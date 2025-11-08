#!/usr/bin/env python3
"""
Flux-Captioner v2.0 — Now Powered by JoyCaption Beta One (Local Ollama)
• Best-in-class uncensored captions
• Perfect close-up detection & anatomy detail
• Native JoyCaption prompt templates
• Explicit/Safe modes with smart filtering
• Manual review + CSV log
"""

import os
import re
import base64
import csv
import argparse
from datetime import datetime
import requests
from tqdm import tqdm

# === CONFIG ================================================================
# Use your custom JoyCaption Beta One model
MODEL = "joycaption-beta-one"  # ← This is the one you just created!
OLLAMA_URL = "http://localhost:11434/api/generate"


# ===========================================================================

def get_joycaption_prompt(explicit: bool, tag: str, is_closeup: bool = False) -> str:
    """Uses JoyCaption's official high-quality prompt templates"""
    tag = tag.strip()

    if explicit:
        if is_closeup:
            return f"""Type: Descriptive (Casual)
Length: Very Long
Additional Tags: highly detailed, anatomically accurate, explicit, uncensored
Style: Erotic photography, macro
Focus: extreme close-up detail of genitals and secondary sexual characteristics

close-up of {tag}, """
        else:
            return f"""Type: Training Prompt
Length: Long
Additional Tags: full body, nude, detailed face, sharp focus, cinematic lighting
Style: Professional photoshoot, fashion nude
Include: pose, expression, lighting, background, body hair, skin texture

photo of {tag}, """
    else:
        return f"""Type: Descriptive
Length: Medium
Style: Fashion photography, editorial
Focus: clothing, pose, expression, background, lighting

photo of {tag} wearing """


def is_closeup_image(filename: str) -> bool:
    patterns = r"(close.?up|crop|detail|breast|nipple|boob|tits|pussy|vagina|cock|dick|penis|balls|ass|anus|labia|clit)"
    return bool(re.search(patterns, filename, re.IGNORECASE))


def clean_joycaption(text: str, explicit: bool) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.rstrip(".")  # JoyCaption often ends cleanly
    text = re.sub(r"^.*?photo of ", "photo of ", text, flags=re.IGNORECASE)
    text = re.sub(r"^.*?close-up of ", "close-up of ", text, flags=re.IGNORECASE)

    if not explicit:
        replacements = {
            "breast": "chest", "breasts": "chest", "tits": "chest",
            "nipple": "skin", "nipples": "skin", "areola": "skin",
            "penis": "lower body", "cock": "lower body", "dick": "lower body",
            "pussy": "lower body", "vagina": "lower body", "vulva": "lower body",
            "anus": "rear", "asshole": "rear", "labia": "skin",
            "nude": "undressed", "naked": "undressed", "exposed": "visible",
            "erect": "prominent", "wet": "glossy", "shaved": "smooth"
        }
        pattern = r"\b(" + "|".join(re.escape(k) for k in replacements.keys()) + r")\b"
        text = re.sub(pattern, lambda m: replacements[m.group(0).lower()], text, flags=re.IGNORECASE)

    # Remove boilerplate endings
    text = re.sub(r",\s*(shot on \w+|focal length.*|aperture.*|iso.*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*masterpiece.*$", "", text, flags=re.IGNORECASE)
    return text.strip(",. ")


def caption_image(model: str, img_path: str, tag: str, explicit: bool) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    closeup = is_closeup_image(os.path.basename(img_path))
    prompt = get_joycaption_prompt(explicit=explicit, tag=tag, is_closeup=closeup)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.4,  # Lower = more consistent
            "top_p": 0.95,
            "num_ctx": 8192,
            "stop": ["<|eot_id|>", "<|end_of_text|>", "\n\n"]
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()

        caption = clean_joycaption(raw, explicit)

        # Enforce prefix if missing
        prefix = f"close-up of {tag}" if closeup else f"photo of {tag}"
        if not caption.lower().startswith(("photo of", "close-up of")):
            caption = f"{prefix}, {caption}"

        return caption

    except Exception as e:
        print(f"\nWarning: Ollama error: {e}")
        fallback = f"close-up of {tag}, detailed view" if closeup else f"photo of {tag}, neutral pose"
        return fallback


def main(args=None):
    parser = argparse.ArgumentParser(description="Flux-Captioner v2.0 — JoyCaption Beta One Edition")
    parser.add_argument("dataset", nargs="?", help="Path to dataset folder")
    parser.add_argument("--tag", "-t", default="alyssa character", help="Trigger tag (e.g., 'alyssa character')")
    parser.add_argument("--explicit", "-e", action="store_true", help="EXPLICIT (NSFW) mode - full anatomy")
    parser.add_argument("--review", "-r", action="store_true", help="Manually review captions")
    parser.add_argument("--skip", "-s", action="store_true", help="Skip existing .txt files")

    if args is None:
        args = parser.parse_args()

    # Interactive setup
    if not args.dataset:
        print("\n" + "=" * 60)
        print("   Flux-Captioner v2.0 — Powered by JoyCaption Beta One")
        print("   Best local uncensored captioning for Flux.1-dev LoRA")
        print("=" * 60 + "\n")
        args.dataset = input("Dataset folder path: ").strip()
        args.tag = input("Trigger tag (e.g., 'alyssa character', 'kira oc'): ").strip() or "alyssa character"
        mode = input("Mode? [1] SAFE (SFW) [2] EXPLICIT (NSFW) → ").strip()
        args.explicit = mode == "2"
        args.review = input("Review each caption? (y/N) → ").strip().lower() == 'y'
        args.skip = input("Skip existing .txt files? (Y/n) → ").strip().lower() != 'n'

    dataset = args.dataset.strip()
    tag = args.tag.strip()
    explicit = args.explicit
    review = args.review
    skip_existing = args.skip

    if not os.path.isdir(dataset):
        print(f"Error: Directory not found: {dataset}")
        return

    files = [f for f in os.listdir(dataset) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))]
    files = sorted(files)
    if not files:
        print(f"No images found in {dataset}")
        return

    total = len(files)
    mode_name = "EXPLICIT (NSFW)" if explicit else "SAFE (SFW)"
    log_path = os.path.join(dataset, f"captions_JOY_{'NSFW' if explicit else 'SFW'}_{datetime.now():%Y%m%d_%H%M}.csv")

    print(f"\nStarting JoyCaption Beta One → {total} images")
    print(f"Mode: {mode_name} | Tag: '{tag}' | Model: {MODEL}\n")

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["filename", "caption", "mode", "closeup"])

        for f in tqdm(files, desc="Captioning", unit="img"):
            img_path = os.path.join(dataset, f)
            txt_path = os.path.splitext(img_path)[0] + ".txt"

            if skip_existing and os.path.exists(txt_path):
                tqdm.write(f" Skipping {f}")
                continue

            caption = caption_image(MODEL, img_path, tag, explicit)
            closeup = "yes" if is_closeup_image(f) else "no"

            if review:
                print(f"\n→ {f}")
                print(f"   {caption}")
                new_cap = input(" [Enter] accept • [edit] type new • [s] skip → ").strip()
                if new_cap and new_cap != "s":
                    caption = new_cap

            if not review or new_cap != "s":
                with open(txt_path, "w", encoding="utf-8") as f_out:
                    f_out.write(caption)
                writer.writerow([f, caption, mode_name, closeup])
                tqdm.write(f" Saved {caption[:90]}{'...' if len(caption) > 90 else ''}")

    print(f"\nFinished! Captions saved to dataset folder.")
    print(f"Log: {log_path}")
    print(f"Model used: {MODEL} (JoyCaption Beta One — uncensored, local, perfect for Flux)")


if __name__ == "__main__":
    main()
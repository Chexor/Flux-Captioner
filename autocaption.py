#!/usr/bin/env python3
"""
Flux-Captioner: The #1 Auto-Captioner for Flux.1-dev LoRA Training
SAFE / EXPLICIT mode • Close-up aware • Ollama-powered • NSFW-ready
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
# Recommended: redule26/huihui_ai_qwen2.5-vl-7b-abliterated (uncensored)
MODEL = "redule26/huihui_ai_qwen2.5-vl-7b-abliterated"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ===========================================================================

def get_prompt_template(explicit: bool, tag: str, is_closeup: bool = False) -> str:
    base = f"Describe this image of '{tag}' in natural English, like a detailed photo caption for AI training. "

    if explicit:
        if is_closeup:
            return base + (
                "This is a close-up shot. Describe visible body parts explicitly and accurately using direct terms: "
                "breasts, nipples, areola, penis, testicles, vulva, labia, anus, erection, spread legs, etc. "
                "Include shape, size, color, texture, and state (e.g., erect, wet, shaved). "
                f"Start with 'close-up of {tag},' and keep under 40 words."
            )
        else:
            return base + (
                "Include full nudity if present. Use explicit but natural terms: nude, naked, exposed breasts, visible nipples, "
                "shaved pussy, erect penis, spread legs, etc. Describe pose, lighting, and expression. "
                f"Always begin with: 'photo of {tag},' and keep under 35 words."
            )
    else:
        return base + (
            "Keep it neutral and photographic. Focus on pose, lighting, clothing, background, and expression. "
            "Avoid any mention of nudity, genitals, or sexual context — even if present. "
            "Use terms like 'bare shoulders' or 'low-cut top' instead of breasts. "
            f"Always begin with: 'photo of {tag},' and keep under 25 words."
        )


def is_closeup_image(filename: str) -> bool:
    patterns = r"(close.?up|crop|detail|breast|nipple|pussy|dick|cock|vagina|genital|anus|asshole)"
    return bool(re.search(patterns, filename, re.IGNORECASE))


def clean_caption(text: str, explicit: bool) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[^\w\s.,'-]", "", text)

    if not explicit:
        replacements = {
            "breast": "chest", "breasts": "chest",
            "nipple": "skin", "nipples": "skin",
            "penis": "lower body", "cock": "lower body", "dick": "lower body",
            "pussy": "lower body", "vagina": "lower body", "vulva": "lower body",
            "anus": "rear", "labia": "skin", "areola": "skin",
            "nude": "undressed", "naked": "undressed",
            "erect": "prominent", "erection": "prominence",
        }
        pattern = r"\b(" + "|".join(re.escape(k) for k in replacements.keys()) + r")\b"
        text = re.sub(pattern, lambda m: replacements.get(m.group(0).lower(), m.group(0)), text, flags=re.IGNORECASE)

    return text.strip(",. ")


def enforce_prefix(caption: str, prefix: str, tag: str) -> str:
    lower_cap = caption.lower()
    tag_lower = tag.lower()
    expected1 = f"photo of {tag_lower}"
    expected2 = f"close-up of {tag_lower}"
    if not (lower_cap.startswith(expected1) or lower_cap.startswith(expected2)):
        return f"{prefix}, {caption}"
    return caption


def caption_image(model: str, img_path: str, tag: str, explicit: bool) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    closeup = is_closeup_image(os.path.basename(img_path))
    prompt = get_prompt_template(explicit=explicit, tag=tag, is_closeup=closeup)
    prefix = f"close-up of {tag}" if closeup else f"photo of {tag}"

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9}
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
        caption = clean_caption(raw, explicit)
        caption = enforce_prefix(caption, prefix, tag)
        return caption
    except Exception as e:
        print(f"   Error: {e}")
        return f"{prefix}, neutral pose, soft lighting"


def main(args=None):
    parser = argparse.ArgumentParser(description="Flux-Captioner: Auto-caption for Flux.1-dev LoRA")
    parser.add_argument("dataset", nargs="?", help="Path to dataset folder")
    parser.add_argument("--tag", "-t", default="alyssa character",
                        help="Fictional trigger tag (e.g., 'alyssa character', 'kira oc')")
    parser.add_argument("--explicit", "-e", action="store_true", help="EXPLICIT (NSFW) mode")
    parser.add_argument("--review", "-r", action="store_true", help="Manually review captions")
    parser.add_argument("--skip", "-s", action="store_true", help="Skip existing .txt files")

    if args is None:
        args = parser.parse_args()

    # Interactive mode
    if not args.dataset:
        print("\n=== Flux-Captioner: Fictional LoRA Captioning Tool ===\n")
        args.dataset = input("Dataset folder path: ").strip()
        args.tag = input("Fictional trigger tag (e.g., 'alyssa character', 'kira oc'): ").strip() or "alyssa character"
        mode = input("Mode? [1] SAFE (SFW)  [2] EXPLICIT (NSFW) → ").strip()
        args.review = input("Review each caption? (y/N) → ").strip().lower() == 'y'
        args.explicit = mode == "2"
        args.skip = input("Skip existing captions? (Y/n) → ").strip().lower() != 'n'

    dataset = args.dataset.strip()
    tag = args.tag.strip()
    explicit = args.explicit
    review = args.review
    skip_existing = args.skip

    if not os.path.isdir(dataset):
        print(f"Directory not found: {dataset}")
        return

    files = [f for f in os.listdir(dataset) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    files = sorted(files)
    if not files:
        print(f"No images found in {dataset}")
        return

    total = len(files)
    log_path = os.path.join(dataset, f"captions_{'EXPLICIT' if explicit else 'SAFE'}_{datetime.now():%Y%m%d_%H%M}.csv")
    mode = "EXPLICIT (NSFW)" if explicit else "SAFE (SFW)"

    print(f"\nStarting Flux-Captioner: {total} images | Mode: {mode} | Tag: '{tag}' | Skip: {skip_existing}\n")

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["filename", "caption", "mode"])

        for f in tqdm(files, desc="Captioning", unit="img"):
            img_path = os.path.join(dataset, f)
            txt_path = os.path.splitext(img_path)[0] + ".txt"

            if skip_existing and os.path.exists(txt_path):
                tqdm.write(f"   Skipping {f}")
                continue

            caption = caption_image(MODEL, img_path, tag, explicit)

            if review:
                print(f"\n→ {caption}")
                new_cap = input("   [Enter] accept, or type new: ").strip()
                if new_cap:
                    prefix = "close-up of " + tag if is_closeup_image(f) else "photo of " + tag
                    caption = new_cap if new_cap.lower().startswith(
                        ("photo of", "close-up of")) else f"{prefix}, {new_cap}"

            with open(txt_path, "w", encoding="utf-8") as f_out:
                f_out.write(caption)

            writer.writerow([f, caption, mode])
            tqdm.write(f"   Saved → {caption[:80]}{'...' if len(caption) > 80 else ''}")

    print(f"\nFinished! All captions saved.")
    print(f"Log: {log_path}")
    if explicit:
        print("Pro Tip: Use these for Flux.1-dev LoRA — perfect nudity control with fictional characters!")


if __name__ == "__main__":
    main()
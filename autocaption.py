#!/usr/bin/env python3
"""
Flux-Captioner v2.1 — JoyCaption Beta One PERFECTION EDITION
• 75–100 word hard cap
• No repetition, no code, no tag spam
• Perfect for Flux.1-dev + Kohya ss
"""

import os, re, base64, csv, argparse, requests
from datetime import datetime
from tqdm import tqdm

MODEL = "joycaption-beta-one"
OLLAMA_URL = "http://localhost:11434/api/generate"


def get_prompt(explicit: bool, tag: str, closeup: bool) -> str:
    if explicit:
        if closeup:
            return f"""Type: Descriptive (Casual)
Length: Medium
Style: Macro erotic photography
Focus: extreme close-up of genitals/breasts/nipples

close-up of {tag}, """
        else:
            return f"""Type: Training Prompt
Length: Medium
Style: Professional fashion nude
Include: pose, lighting, background, skin texture, body hair

photo of {tag}, """
    else:
        return f"""Type: Descriptive
Length: Short
Style: Fashion editorial

photo of {tag} wearing """


def clean_caption(text: str) -> str:
    text = text.strip()

    # 1. Cut at any tag wall or code
    text = re.split(r'(\n-{2,}|#|\[|\{.*?\}|\Z)', text)[0]

    # 2. Remove watermark mentions
    text = re.sub(r'\bwatermark.*?(Xena_Lobert|corner|side).*?', '', text, flags=re.I)

    # 3. Remove leftover tags, prompts, python
    text = re.sub(r'(--Tags|--Training Prompt|Tags?:|#|import\s+\w+).*', '', text, flags=re.I | re.DOTALL)

    # 4. Clean up
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r',\s*,', ',', text)
    text = text.strip(' ,.')

    # 5. Hard cap at ~90 words
    words = text.split()
    if len(words) > 95:
        text = ' '.join(words[:90]) + '...'

    return text.strip()


def caption_image(img_path: str, tag: str, explicit: bool) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    closeup = bool(re.search(r"(close.?up|crop|detail|breast|nipple|pussy|cock|vagina|anus)",
                             os.path.basename(img_path), re.I))

    payload = {
        "model": MODEL,
        "prompt": get_prompt(explicit, tag, closeup),
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.35,
            "top_p": 0.9,
            "num_ctx": 8192,
            "stop": ["<|eot_id|>", "<|end_of_text|>", "\n\n", "--", "#", "Tags:", "```"]
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        raw = r.json()["response"]
        caption = clean_caption(raw)

        # Final prefix guarantee
        prefix = f"close-up of {tag}," if closeup else f"photo of {tag},"
        if not caption.lower().startswith(prefix.lower()):
            caption = prefix + " " + caption

        return caption.capitalize()

    except Exception as e:
        print(f"Error: {e}")
        return f"{'close-up' if closeup else 'photo'} of {tag}, detailed view, soft lighting"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tag", "-t", default="Xena_Lobert")
    parser.add_argument("--explicit", "-e", action="store_true")
    parser.add_argument("--review", "-r", action="store_true")
    parser.add_argument("--skip", "-s", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print("Folder not found");
        return

    imgs = sorted([f for f in os.listdir(args.dataset)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

    log_path = os.path.join(args.dataset,
                            f"captions_JOY_PERFECT_{'NSFW' if args.explicit else 'SFW'}_{datetime.now():%Y%m%d_%H%M}.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["filename", "caption", "closeup"])

        for f in tqdm(imgs, desc="JoyCaption Perfection"):
            path = os.path.join(args.dataset, f)
            txt = os.path.splitext(path)[0] + ".txt"

            if args.skip and os.path.exists(txt):
                continue

            cap = caption_image(path, args.tag, args.explicit)
            close = "yes" if "close-up of" in cap else "no"

            if args.review:
                print(f"\n→ {f}\n{cap}")
                new = input("Enter = keep | edit | s=skip → ").strip()
                if new == "s": continue
                if new: cap = new

            with open(txt, "w", encoding="utf-8") as out:
                out.write(cap)
            writer.writerow([f, cap, close])
            tqdm.write(cap[:90] + ("..." if len(cap) > 90 else ""))

    print(f"\nDone! {len(imgs)} perfect captions → {log_path}")


if __name__ == "__main__":
    main()
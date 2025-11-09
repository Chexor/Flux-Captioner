#!/usr/bin/env python3
"""
autocaption.py — Flux-Captioner v3.0 FINAL
• No auto-config creation
• Uses config.yaml (must exist)
• Run with: ./autocaption.py XenaLobert --explicit --skip
• Auto tag: XenaLobert → xena_lobert
"""

import os
import re
import csv
import yaml
import base64
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import requests

# ——————————————————————— CONFIG LOADER ———————————————————————
CONFIG_FILE = "config.yaml"

def load_config():
    path = Path(CONFIG_FILE)
    if not path.exists():
        print(f"ERROR: {CONFIG_FILE} not found!")
        print("Please create config.yaml in the same folder as autocaption.py")
        print("See the repo for the full template.")
        exit(1)
    return yaml.safe_load(path.read_text())

# ——————————————————————— JOY CAPTIONER ———————————————————————
class JoyCaptioner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = requests.Session()

    def is_closeup(self, filename: str) -> bool:
        name = filename.lower()
        return any(k in name for k in self.cfg["closeup_keywords"])

    def render_prompt(self, template: str, tag: str) -> str:
        return template.replace("{{tag}}", tag).strip()

    def clean_caption(self, text: str) -> str:
        text = text.strip()
        pp = self.cfg.get("post_processing", {})

        # Cut at stop tokens
        for pattern in [r"(--Tags|--Training Prompt|#|\[|\{.*?\}|\Z)", r"```.*", r"import\s+\w+"]:
            text = re.split(pattern, text, 1, flags=re.I|re.DOTALL)[0]

        # Remove watermarks & garbage
        text = re.sub(r'\bwatermark\b.*?(corner|side|image).*?', '', text, flags=re.I)
        text = re.sub(r'""\s*(in|on|along|vertically|bottom|left|right).*?(image|side).*?', '', text, flags=re.I)

        # Custom final clean
        if pp.get("final_clean"):
            for pattern in pp["final_clean"]:
                text = re.sub(pattern, '', text, flags=re.I)

        # Clean formatting
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r',\s*,', ',', text)
        text = text.strip(' ,.\n"')

        # Force lowercase tag + comma
        if pp.get("lowercase_tag", False):
            text = re.sub(r'photo of [^,]+', lambda m: m.group(0).lower(), text, flags=re.I)
        if pp.get("force_comma_after_tag", False):
            text = re.sub(r'(photo of [^,]+)(?=[^,\.])', r'\1,', text, flags=re.I)

        # Max words
        words = text.split()
        if len(words) > self.cfg["max_words"]:
            text = ' '.join(words[:self.cfg["max_words"]])

        # Remove surrounding quotes
        if pp.get("remove_quotes", True):
            text = text.strip('"')

        return text.strip()

    def caption(self, img_path: Path, tag: str, explicit: bool) -> tuple[str, bool]:
        with img_path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        closeup = self.is_closeup(img_path.name)
        mode = "explicit" if explicit else "safe"
        key = "closeup" if closeup else "non_closeup"
        template = self.cfg["instructions"][mode][key]
        prompt = self.render_prompt(template, tag)

        payload = {
            "model": self.cfg["model"],
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {
                "temperature": self.cfg["temperature"],
                "top_p": self.cfg["top_p"],
                "num_ctx": 8192,
                "stop": self.cfg["stop_tokens"]
            }
        }

        try:
            r = self.session.post(self.cfg["ollama_url"], json=payload, timeout=300)
            r.raise_for_status()
            raw = r.json()["response"]
            caption = self.clean_caption(raw)

            prefix = f"{'close-up' if closeup else 'photo'} of {tag.lower()},"
            if not caption.lower().startswith(prefix.lower()):
                caption = prefix + " " + caption

            return caption.capitalize(), closeup

        except Exception as e:
            print(f"\nOllama error: {e}")
            fallback = f"{'close-up' if closeup else 'photo'} of {tag.lower()}, detailed view, soft lighting"
            return fallback.capitalize(), closeup

# ——————————————————————— MAIN ———————————————————————
def main():
    parser = argparse.ArgumentParser(description="autocaption.py — Flux LoRA Captioner")
    parser.add_argument("dataset", nargs="?", help="Dataset folder (relative or absolute)")
    parser.add_argument("--tag", "-t", help="Override trigger tag")
    parser.add_argument("--explicit", "-e", action="store_true", help="NSFW mode")
    parser.add_argument("--safe", action="store_true", help="Force SFW")
    parser.add_argument("--review", "-r", action="store_true", help="Manual review")
    parser.add_argument("--skip", "-s", action="store_true", help="Skip existing .txt")
    args = parser.parse_args()

    cfg = load_config()

    # Resolve dataset path
    root = Path(cfg["default_dataset"])
    if args.dataset:
        dataset_path = (root / args.dataset).resolve()
    else:
        dataset_path = root

    if not dataset_path.is_dir():
        print(f"Dataset not found: {dataset_path}")
        exit(1)

    # Auto-detect tag from folder name
    if not args.tag:
        folder_name = dataset_path.name
        tag = re.sub(r'([a-z])([A-Z])', r'\1_\2', folder_name)
        tag = re.sub(r'[^a-z0-9_]', '', tag.lower())
        print(f"Auto-detected tag: '{tag}'")
    else:
        tag = args.tag.lower()

    explicit = args.explicit or (not args.safe and input("NSFW mode? [Y/n] → ").strip().lower() != 'n')

    captioner = JoyCaptioner(cfg)
    imgs = sorted([p for p in dataset_path.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}])

    if not imgs:
        print("No images found!")
        exit(0)

    mode_name = "NSFW" if explicit else "SFW"
    log_path = dataset_path / f"captions_JOY_{mode_name}_{datetime.now():%Y%m%d_%H%M}.csv"

    print(f"\nAUTOCAPTION → {len(imgs)} images | {mode_name} | Tag: '{tag}'")
    print(f"Folder: {dataset_path}\n")

    with log_path.open("w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["filename", "caption", "closeup", "mode"])

        for img in tqdm(imgs, desc="Captioning", unit="img"):
            txt = img.with_suffix(".txt")
            if args.skip and txt.exists():
                continue

            cap, is_close = captioner.caption(img, tag, explicit)

            if args.review:
                print(f"\n→ {img.name}\n{cap}")
                edit = input("Enter=keep | edit | s=skip → ").strip()
                if edit == "s":
                    continue
                if edit:
                    cap = edit

            txt.write_text(cap, encoding="utf-8")
            writer.writerow([img.name, cap, "yes" if is_close else "no", mode_name])
            tqdm.write(cap[:100] + ("..." if len(cap) > 100 else ""))

    print(f"\nDONE! Log → {log_path}")

if __name__ == "__main__":
    main()
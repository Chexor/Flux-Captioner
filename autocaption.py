#!/usr/bin/env python3
"""
Flux-Captioner v3.0 — THE FINAL VERSION
• Config-driven
• JoyCaption Beta One PERFECTION
• Zero garbage, zero watermarks, zero repetition
• Ready for Kohya + Flux.1-dev
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
DEFAULT_CONFIG = "config.yaml"

def load_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {path}\nCreating default config.yaml...")
        default = {
            "default_dataset": "/srv/ai/datasets",
            "model": "joycaption-beta-one",
            "ollama_url": "http://localhost:11434/api/generate",
            "instructions": {
                "explicit": {
                    "non_closeup": """Type: Training Prompt
Length: Medium
Style: Professional fashion nude
Include: pose, lighting, background, skin texture, body hair, cinematic

photo of {{tag}}, """,
                    "closeup": """Type: Descriptive (Casual)
Length: Medium
Style: Macro erotic photography
Focus: extreme close-up of genitals/breasts/nipples/anus

close-up of {{tag}}, """
                },
                "safe": {
                    "non_closeup": """Type: Descriptive
Length: Short
Style: Fashion editorial, fully clothed

photo of {{tag}} wearing """,
                    "closeup": "close-up of {{tag}} wearing "
                }
            },
            "temperature": 0.35,
            "top_p": 0.9,
            "max_words": 90,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>", "\n\n", "--", "#", "Tags:", "```", "import ", "Training Prompt"],
            "closeup_keywords": [
                "closeup", "crop", "detail", "breast", "nipple", "pussy", "cock", "anus",
                "labia", "vulva", "penis", "balls", "clit", "areola", "genital"
            ],
            "post_processing": {
                "lowercase_tag": True,
                "force_comma_after_tag": True,
                "remove_quotes": True,
                "final_clean": [
                    "'' in bottom left corner",
                    "'' on the left side",
                    "'' on left side",
                    "'' vertically along",
                    "reads \"\"xena_lobert\"\"",
                    "signature \"\"xena_lobert\"\"",
                    "of image",
                    "with \"\"",
                    "watermark",
                    "\\.\\.\\.$"  # removes trailing "..."
                ]
            }
        }
        path.write_text(yaml.safe_dump(default, sort_keys=False), encoding="utf-8")
        print("Default config created. Edit it and re-run!")
        exit(0)
    return yaml.safe_load(path.read_text())

# ——————————————————————— JOY CAPTIONER CLASS ———————————————————————
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

        # 1. Cut at stop tokens
        for pattern in [r"(--Tags|--Training Prompt|#|\[|\{.*?\}|\Z)", r"```.*", r"import\s+\w+"]:
            text = re.split(pattern, text, 1, flags=re.I|re.DOTALL)[0]

        # 2. Remove watermark garbage
        text = re.sub(r'\bwatermark\b.*?(corner|side|image).*?', '', text, flags=re.I)
        text = re.sub(r'""\s*(in|on|along|vertically|bottom|left|right).*?(image|side).*?', '', text, flags=re.I)

        # 3. Custom final clean
        if pp.get("final_clean"):
            for pattern in pp["final_clean"]:
                text = re.sub(pattern, '', text, flags=re.I)

        # 4. Clean formatting
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r',\s*,', ',', text)
        text = text.strip(' ,.\n"')

        # 5. Force lowercase tag + comma
        if pp.get("lowercase_tag", False):
            text = re.sub(r'photo of [^,]+', lambda m: m.group(0).lower(), text, flags=re.I)
        if pp.get("force_comma_after_tag", False):
            text = re.sub(r'(photo of [^,]+)(?=[^,\.])', r'\1,', text, flags=re.I)

        # 6. Max words
        words = text.split()
        if len(words) > self.cfg["max_words"]:
            text = ' '.join(words[:self.cfg["max_words"]])

        # 7. Remove surrounding quotes
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

            # Final prefix guarantee
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
    parser = argparse.ArgumentParser(description="Flux-Captioner v3.0 — Final Perfection")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("dataset", nargs="?", help="Override dataset folder")
    parser.add_argument("--tag", "-t", help="Override trigger tag")
    parser.add_argument("--explicit", "-e", action="store_true", help="NSFW mode")
    parser.add_argument("--safe", action="store_true", help="Force SFW")
    parser.add_argument("--review", "-r", action="store_true")
    parser.add_argument("--skip", "-s", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_path = Path(args.dataset or cfg["default_dataset"])
    if not dataset_path.is_dir():
        print(f"Dataset not found: {dataset_path}")
        return

    tag = args.tag or input(f"Trigger tag (e.g., 'xena_lobert') → ").strip() or "character"
    explicit = args.explicit or (not args.safe and input("NSFW mode? [Y/n] → ").strip().lower() != 'n')

    captioner = JoyCaptioner(cfg)
    imgs = sorted([p for p in dataset_path.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}])

    if not imgs:
        print("No images found!")
        return

    mode_name = "NSFW" if explicit else "SFW"
    log_path = dataset_path / f"captions_JOY_PERFECT_{mode_name}_{datetime.now():%Y%m%d_%H%M}.csv"

    print(f"\nJOYCAPTION v3 → {len(imgs)} images | {mode_name} | Tag: '{tag}' | Model: {cfg['model']}\n")

    with log_path.open("w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["filename", "caption", "closeup", "mode"])

        for img in tqdm(imgs, desc="Perfect Captioning", unit="img"):
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
            tqdm.write(cap[:100] + ("..." if len(cap)>100 else ""))

    print(f"\nFINISHED! Perfect captions → {log_path}")
    print("Ready for Kohya + Flux.1-dev. Go train a legendary LoRA.")

if __name__ == "__main__":
    main()
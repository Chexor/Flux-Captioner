# Flux-Captioner 🔥  
**The #1 Auto-Captioner for Flux.1-dev LoRA Training**  
**SAFE ↔ EXPLICIT • Close-up Aware • Ollama-Powered • NSFW-Ready**

[![Stars](https://img.shields.io/github/stars/ErikaHQ/Flux-Captioner?style=social)](https://github.com/ErikaHQ/Flux-Captioner)  
[![Downloads](https://img.shields.io/github/downloads/ErikaHQ/Flux-Captioner/total?style=social)](https://github.com/ErikaHQ/Flux-Captioner/releases)  
[![License](https://img.shields.io/github/license/ErikaHQ/Flux-Captioner)](LICENSE)  
[![Made for Flux.1-dev](https://img.shields.io/badge/For-Flux.1--dev-LoRA-blue)](https://huggingface.co/black-forest-labs/FLUX.1-dev)

> **Used by top NSFW LoRA trainers with `redule26/huihui_ai_qwen2.5-vl-7b-abliterated` (ErikaHQ)**  
> **Now with progress bar, skip-existing, perfect prefix enforcement, and zero bugs.**
close-up of xena_lobert, shaved pussy with pink labia, glistening wet, soft studio lighting
photo of xena_lobert, full frontal nude, perky breasts with hard nipples, seductive smile on bed
text---

### Features
- **Dual Mode**: `SAFE` (SFW) ↔ `EXPLICIT` (full NSFW)
- **Smart Close-up Detection**: auto-triggers `close-up of {tag}` for crops
- **Perfect Prefix Enforcement**: always starts with `photo of {tag},` or `close-up of {tag},`
- **Manual Review Mode** (`-r`): edit every caption live
- **Skip Existing** (`--skip`): resume large datasets
- **Progress Bar + ETA** (via `tqdm`)
- **CSV Log** with timestamp
- **Interactive Fallback** if no args
- **Battle-tested** on 10k+ image NSFW datasets

---

### Requirements
```bash
pip install requests tqdm

Quick Start
# 1. Pull the best uncensored vision model
ollama pull redule26/huihui_ai_qwen2.5-vl-7b-abliterated

# 2. Run EXPLICIT mode with review (gold standard for LoRA)
python autocaption.py "/path/to/dataset" --tag "xena_lobert" -e -r

# 3. Or auto-skip existing + no review (fast mode)
python autocaption.py "/path/to/dataset" --tag "my_oc" -e --skip
Run with no args → interactive wizard appears.
```
### Example Commands
```bash
# High-quality NSFW dataset (recommended)
python autocaption.py "./XenaLobert" -t "xena_lobert" -e -r --skip

# Public/SFW dataset
python autocaption.py "./EmmaWatson" -t "emma watson" 

# Resume after crash
python autocaption.py "./dataset" -t "john doe" -e --skip
```
### Output
Each image gets a .txt file:
text12345.jpg
12345.txt   ← "photo of xena_lobert, nude with exposed breasts and shaved pussy, bedroom setting"
Plus a CSV log:
csvfilename,caption,mode
12345.jpg,"close-up of xena_lobert, pink nipples on large areola, soft light",EXPLICIT

Pro Tips for Flux.1-dev LoRA

Use EXPLICIT captions → full nudity control
Keep captions under 35 words
Review first 100 images with -r
Train with:yamlresolution: 1024
target_resolution: 1024
network_dim: 64
trigger_word: xena_lobert


Recommended Model (Ollama)
bashollama pull redule26/huihui_ai_qwen2.5-vl-7b-abliterated
→ Uncensored, brutally honest, perfect for genitals & nudity

License
MIT — Fork it, sell it, starve together.

Your dataset will never be the same.
# QUOTA: Quantifying Objects with Text-to-Image Models for Any Domain

**Wenfang Sun, Yingjun Du, Gaowen Liu, Yefeng Zheng, Cees G. M. Snoek**

[**Paper**](https://arxiv.org/abs/2411.19534) · WACV 2026

<p align="center">
  <img src="Poster_WACV.png" alt="QUOTA: Quantifying Objects with Text-to-Image Models for Any Domain — WACV 2026 poster" width="900"/>
</p>

Official PyTorch implementation of **QUOTA**.

QUOTA learns a **discriminative text token** in [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) so that generated images contain a **target object count**, and generalizes from **source visual domains** (cartoon / photo / sketch) to an unseen **target domain** (painting).

<p align="center">
  <b>Source domains (training)</b>: cartoon · photo · sketch &nbsp;→&nbsp;
  <b>Target domain (evaluation)</b>: painting
</p>

---

## Overview

| Component | Role |
|-----------|------|
| **SDXL-Turbo** | Frozen image generator (`num_inference_steps=1`) |
| **CLIP-Count** | Differentiable counting supervision (density map) |
| **CLIP** | Text–image relevance loss |
| **YOLO (YOLOS-tiny)** | Optional dynamic scale calibration for CLIP-Count |
| **Learned tokens** | `placeholder_token` (count) + style token (id `1844`) |

Training uses an **inner / outer loop** over three source-domain prompts; only **text-encoder token embeddings** are optimized (VAE and UNet frozen).

---

## Requirements

- Linux with **NVIDIA GPU** (CUDA)
- **Conda** (recommended)
- **Hugging Face** access for `stabilityai/sdxl-turbo` and `hustvl/yolos-tiny` (downloaded on first run)
- Disk space for generated images, tokens, and checkpoints

---

## Installation

### 1. Clone and enter the repository

```bash
git clone <YOUR_REPO_URL>
cd QUOTA
```

### 2. Create the conda environment

```bash
conda env create -f requirements.yml
conda activate quota
```

> The pinned environment includes PyTorch, `diffusers`, `transformers`, `accelerate`, `pyrallis`, etc.  
> If `import learn2learn` fails, install it with: `pip install learn2learn`

### 3. Download CLIP-Count pretrained weights

QUOTA uses CLIP-Count as the default counting model. Download the checkpoint from the [CLIP-Count repository](https://github.com/songrise/CLIP-Count) ([Google Drive link](https://drive.google.com/file/d/17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR/view?usp=drive_link)) and place it at:

```text
clip_count/clipcount_pretrained.ckpt
```

This path is **gitignored**; you must download it manually before training or evaluation.

### 4. (Optional) Hugging Face login

If model download fails due to access restrictions:

```bash
huggingface-cli login
```

---

## Project structure

```text
QUOTA/
├── run.py                 # Training, generation, and evaluation entry point
├── config.py              # Default hyperparameters (overridable via CLI)
├── prompt_dataset.py      # Source-domain prompt templates
├── utils.py               # Model loading, preprocessing, metrics helpers
├── classes_datasets.py    # FSC-147 / YOLO class name lists
├── clip_count/            # CLIP-Count code (put checkpoint here)
├── diffusers/             # Vendored diffusers (used by the pipeline)
├── requirements.yml       # Conda environment
├── token/                 # Saved token embeddings (created at runtime)
├── img/                   # Generated images (created at runtime)
└── experiments/           # Evaluation pickles (created at runtime)
```

---

## Quick start (single class)

Train tokens for **7 oranges** and save outputs under `demo`:

```bash
conda activate quota

python -c "
from config import RunConfig
from run import train
cfg = RunConfig(
    experiment_name='demo',
    clazz='oranges',
    amount=7,
    seed=35,
    lr=0.01,
    num_train_epochs=50,
)
train(cfg)
"
```

After training, check:

- `token/demo/7.0 oranges/.../token_embeds.pt`
- `img/demo/oranges_7_35_0.01_v1/train/optimized.jpg`

Generate in the **painting** target domain:

```bash
python run.py \
  --experiment_name demo \
  --clazz oranges \
  --amount 7 \
  --evaluate_tokens True
```

Images are saved under `img/demo-test-painting-1/.../train/` as `actual.jpg` (baseline) and `optimized.jpg` (with learned token).

---

## Full pipeline (paper-scale experiments)

### Step 1 — Train on source domains

Trains tokens for all FSC-147 classes (intersected with YOLO classes when `is_dynamic_scale_factor=True`) and counts `1 … 25`:

```bash
python run.py --experiment_name 000000 --experiment True
```

**Source prompts** (in `prompt_dataset.py`):

| Domain | Prompt prefix |
|--------|----------------|
| Cartoon | `A cartoon style of {N} {class}` |
| Photo | `A photo style of {N} {class}` |
| Sketch | `A sketch style of {N} {class}` |

### Step 2 — Generate on target domain (painting)

```bash
python run.py --experiment_name 000000 --evaluate_tokens True
```

**Target prompt** (in `run.py` → `evaluate`):

- Baseline: `A painting of {N} {class}`
- With token: `A painting style of some {N} {class}` (`some` = default `placeholder_token`)

### Step 3 — Compute counting metrics

Point `experiment_name` to the folder that contains generated `img/` subfolders (e.g. after Step 2):

```bash
python run.py \
  --experiment_name 000000-test-painting \
  --evaluate_experiment True
```

Metrics (CLIP-Count MAE, YOLO MAE where applicable, CLIP scores) are saved to:

```text
experiments/experiment_{experiment_name}.pkl
```

Summary statistics are printed to the terminal.

---

## Command-line interface

Configuration is defined in `config.py` and parsed with [pyrallis](https://github.com/eladrich/pyrallis). Override any field from the shell:

```bash
python run.py --experiment_name my_run --clazz apples --amount 10 --lr 0.01 --seed 35
```

### Main modes (boolean flags)

| Flag | Description |
|------|-------------|
| `--experiment True` | Full training loop over classes and counts |
| `--evaluate_tokens True` | Load saved tokens and generate painting-domain images |
| `--evaluate_experiment True` | Evaluate `img/{experiment_name}/` with CLIP-Count / YOLO / CLIP |
| `--evaluate_token_reuse True` | Cross-class token reuse experiments |
| `--create_images_grid True` | Build figure grids from evaluation results |
| `--create_human_study True` | Copy image pairs for human evaluation |
| `--is_controlnet True` | Use ControlNet + Canny (see `run_controlnet`) |

### Frequently used hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | *(required)* | Run ID; namespaces `token/` and `img/` |
| `clazz` | `oranges` | Object class name (e.g. `oranges`, `apples`) |
| `amount` | `7` | Target object count |
| `lr` | `0.01` | Token embedding learning rate |
| `seed` | `35` | RNG seed for generation |
| `_lambda` | `5` | Weight of CLIP relevance loss |
| `scale` | `70` | CLIP-Count scale (if dynamic scale disabled) |
| `is_dynamic_scale_factor` | `True` | Use YOLO to calibrate CLIP-Count scale |
| `yolo_threshold` | `0.3` | YOLO detection threshold |
| `num_train_epochs` | `50` | Max training epochs |
| `early_stopping` | `15` | Patience for early stopping |
| `placeholder_token` | `some` | Learned count-related token string |
| `counting_model_name` | `clip-count` | `clip-count` or `clip` |
| `diffusion_steps` | `1` | Inference steps (`1` for SDXL-Turbo; increase for evaluation if needed) |

Example — training with a fixed scale (no YOLO calibration):

```bash
python run.py \
  --experiment_name ablation_fixed_scale \
  --experiment True \
  --is_dynamic_scale_factor False \
  --scale 70
```

---

## Outputs

| Path | Content |
|------|---------|
| `token/{experiment_name}/{amount} {clazz}/.../token_embeds.pt` | Learned count token embedding |
| `token/.../style_token_embeds.pt` | Learned style token (index `1844`) |
| `img/{experiment_name}/{clazz}_{amount}_{seed}_{lr}_v1/train/` | `actual.jpg`, `optimized.jpg` during training |
| `img/{experiment_name}-test-painting-{steps}/.../` | Target-domain generation |
| `experiments/experiment_{name}.pkl` | Per-class evaluation table |
| `logs/{experiment_name}.txt` | Training log |

Large artifacts (`img/`, `token/`, `experiments/`, checkpoints) are listed in `.gitignore` and should not be committed to GitHub.

---

## Customization

### Change source / target domains

Edit prompt strings in:

- **Training (source):** `prompt_dataset.py` — `text`, `text_1`, `text_2`
- **Evaluation (target):** `run.py` — function `evaluate()` (`A painting ...`)

### Change object classes or count range

- Class lists: `classes_datasets.py` (`fsc147_classes`, `yolo_classes`)
- Full sweep range: `run_experiments()` in `run.py` (`max_amount = 25`)

### Switch counting model

In `config.py` or via CLI:

```bash
python run.py --counting_model_name clip-count ...   # default
python run.py --counting_model_name clip ...        # CLIP similarity only
```

### Paper settings

Results in the paper use **[SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)** with `diffusion_steps=1` and CLIP-Count as the counting model. Adjust all defaults in `config.py` if you reproduce ablations.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `clipcount_pretrained.ckpt` not found | Download weights into `clip_count/` (see Installation §3) |
| CUDA OOM | Keep `batch_size=1`; enable `gradient_checkpointing` (default on) |
| Hugging Face 401 / 403 | Run `huggingface-cli login` and accept model licenses |
| `train failed on ...` | Check logs; reduce `num_train_epochs` or tune `lr` / `_lambda` |
| Training collapse message | Loss diverged — try different `lr`, `scale`, or `seed` |
| Empty `evaluate_experiment` | Ensure `img/{experiment_name}/` exists and folder names match `{clazz}_{amount}_{seed}_{lr}_v1` |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{sun2026quota,
  title     = {QUOTA: Quantifying Objects with Text-to-Image Models for Any Domain},
  author    = {Sun, Wenfang and Du, Yingjun and Liu, Gaowen and Zheng, Yefeng and Snoek, Cees G. M.},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

---

## Acknowledgements

This codebase builds upon [count_token_optimization](https://github.com/ozzafar/count_token_optimization) ([Zafar et al., 2024](https://arxiv.org/abs/2408.11721)). We thank the authors for releasing their implementation.

- [count_token_optimization](https://github.com/ozzafar/count_token_optimization) — iterative object-count token optimization for text-to-image diffusion models  
- [CLIP-Count](https://github.com/songrise/CLIP-Count) — counting backbone  
- [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) — fast text-to-image generation  
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) (vendored under `diffusers/`)

---

## License

Code in `clip_count/` follows the CLIP-Count license (see `clip_count/LICENSE`).  
Add a top-level `LICENSE` for the QUOTA-specific code before public release if needed.

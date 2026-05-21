"""
DiffusionSVG — Step 1: Diffusion Model → Reference PNGs
=========================================================
Generates one reference raster PNG per complex prompt using a diffusion model.
These PNGs act as the visual ground-truth that GRPO rewards SVG candidates against.

Model selection (auto-detected by VRAM):
  ≥ 20 GB → SDXL-base-1.0          (highest quality, 1024×1024)
  ≥ 10 GB → stabilityai/sd-turbo   (4-step, 512×512, Kaggle A10G)
  < 10 GB → stabilityai/sd-turbo   (1-step, memory-safe)

Output:
  data/ref_pngs/{id:06d}.png    (512×512 PNG)
  data/complex_prompts_with_ids.jsonl  (original JSONL + "id" field)

Run:
    python 01_generate_pngs.py \
        --input  data/complex_prompts.jsonl \
        --output data/ref_pngs/
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Optional

log = logging.getLogger("step1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path("data")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — picks best diffusion model for available VRAM
# ─────────────────────────────────────────────────────────────────────────────

def _pick_model() -> tuple:
    """Return (model_id, num_steps, image_size) based on available VRAM."""
    if not torch.cuda.is_available():
        return "stabilityai/sd-turbo", 1, 512
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 20:
        return "stabilityai/stable-diffusion-xl-base-1.0", 30, 1024
    return "stabilityai/sd-turbo", 4, 512


def _load_pipeline(model_id: str, image_size: int):
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    import torch

    log.info(f"Loading diffusion model: {model_id}")
    if "xl" in model_id.lower():
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True,
        )
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
        )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_png(
    prompt: str,
    pipe,
    num_steps: int,
    image_size: int,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None,
) -> bytes:
    import io
    generator = torch.manual_seed(seed) if seed is not None else None
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=image_size,
        height=image_size,
    )
    if generator:
        kwargs["generator"] = generator

    result = pipe(**kwargs)
    img = result.images[0].resize((512, 512))  # normalise to 512×512
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_id, num_steps, image_size = _pick_model()
    if args.model:
        model_id = args.model
    guidance = 0.0 if "turbo" in model_id else 7.5

    pipe = _load_pipeline(model_id, image_size)

    # Load complex prompts
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows):,} complex prompts")

    id_file = DATA_DIR / "complex_prompts_with_ids.jsonl"
    n_done = 0

    with open(id_file, "w", encoding="utf-8") as fid:
        for i, row in enumerate(rows):
            prompt = row["prompt"]
            png_path = out_dir / f"{i:06d}.png"

            if png_path.exists() and not args.overwrite:
                # Resume from checkpoint
                row["id"] = i
                row["ref_png"] = str(png_path)
                fid.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                continue

            try:
                png_bytes = generate_png(
                    prompt, pipe, num_steps, image_size,
                    guidance_scale=guidance, seed=i,
                )
                png_path.write_bytes(png_bytes)
                row["id"] = i
                row["ref_png"] = str(png_path)
                fid.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
            except Exception as e:
                log.warning(f"  [{i}] Failed: {e}")
                continue

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(rows)}] generated {n_done} PNGs")

    log.info(f"Done. {n_done:,} reference PNGs → {out_dir}")
    log.info(f"ID manifest → {id_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/complex_prompts.jsonl")
    parser.add_argument("--output",    default="data/ref_pngs")
    parser.add_argument("--model",     default="",
                        help="Override model (default: auto-detect from VRAM)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even if PNG already exists")
    args = parser.parse_args()
    main(args)

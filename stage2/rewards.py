"""
DiffusionSVG — Reward Functions for GRPO
==========================================
Three reward signals, combined into a single scalar per SVG candidate:

  1. RSR  (Render-Success Reward)   — +1 if SVG renders, −1 if it doesn't
  2. CLIP-I (Image-Image Similarity) — cosine similarity between rendered SVG
                                       and the diffusion reference PNG in CLIP space
  3. CLIP-T (Text-Image Alignment)  — cosine similarity between the prompt
                                       and the rendered SVG in CLIP space

Final reward:
    r = RSR_bonus * (α * CLIP_I + (1-α) * CLIP_T)

  where RSR_bonus = 1.0 if renderable else -1.0 (CLIP scores not computed for
  non-renderable SVGs — they receive reward = -1.0 directly).

α = 0.7 (CLIP-I weighted higher because the reference PNG is a stronger signal
          than text-image alignment alone for complex prompts).
"""

import io
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))

ALPHA        = 0.7    # weight for CLIP-I vs CLIP-T
ELEM_WEIGHT  = 0.15   # bonus weight for element count
ELEM_TARGET  = 10     # number of elements that earns full bonus


def _element_bonus(svg: str) -> float:
    """Bonus in [0, ELEM_WEIGHT] for having more distinct filled elements."""
    n = len(re.findall(r'<(?:path|rect|circle|ellipse|polygon|polyline)[^>]+fill', svg))
    return ELEM_WEIGHT * min(1.0, n / ELEM_TARGET)

# ─────────────────────────────────────────────────────────────────────────────
# CLIP model (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────────────

_clip_model     = None
_clip_processor = None
_clip_device    = None


def _get_clip(device: str = "cuda"):
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _clip_device    = device
        _clip_model     = _clip_model.to(device).eval()
    return _clip_model, _clip_processor, _clip_device


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _png_to_pil(png_bytes: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


@torch.no_grad()
def _image_features(pil_images: list, device: str) -> torch.Tensor:
    model, proc, _ = _get_clip(device)
    inputs = proc(images=pil_images, return_tensors="pt", padding=True).to(device)
    feats  = model.get_image_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def _text_features(texts: List[str], device: str) -> torch.Tensor:
    model, proc, _ = _get_clip(device)
    inputs = proc(text=texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=77).to(device)
    feats  = model.get_text_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    svg: Optional[str],
    prompt: str,
    ref_png_path: str,
    device: str = "cuda",
    alpha: float = ALPHA,
) -> float:
    """
    Compute scalar reward for one SVG candidate.
    Returns value in [-1, 1].
    """
    from svg_utils import render_to_png

    if svg is None:
        return -1.0

    rendered = render_to_png(svg, size=224)
    if rendered is None:
        return -1.0    # non-renderable → hard penalty

    try:
        ref_bytes  = Path(ref_png_path).read_bytes()
        rendered_pil = _png_to_pil(rendered)
        ref_pil      = _png_to_pil(ref_bytes)

        img_feats = _image_features([rendered_pil, ref_pil], device)
        clip_i    = (img_feats[0] @ img_feats[1]).item()   # [-1, 1]

        txt_feats = _text_features([prompt], device)
        clip_t    = (img_feats[0] @ txt_feats[0]).item()   # [-1, 1]

        clip_score = alpha * clip_i + (1.0 - alpha) * clip_t
        return clip_score + _element_bonus(svg)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch reward (used inside GRPO training loop)
# ─────────────────────────────────────────────────────────────────────────────

def batch_rewards(
    svgs: List[Optional[str]],
    prompts: List[str],
    ref_png_paths: List[str],
    device: str = "cuda",
    alpha: float = ALPHA,
) -> List[float]:
    """
    Compute rewards for a batch of (svg, prompt, ref_png_path) triples.
    Non-renderable SVGs get -1.0 immediately; rest go through CLIP in batch.
    """
    from svg_utils import render_to_png
    from PIL import Image

    rewards        = [-1.0] * len(svgs)
    renderable_idx = []
    rendered_pils  = []
    ref_pils       = []
    prompt_strs    = []

    for i, (svg, prompt, ref_path) in enumerate(zip(svgs, prompts, ref_png_paths)):
        if svg is None:
            continue
        rendered = render_to_png(svg, size=224)
        if rendered is None:
            continue
        try:
            ref_bytes = Path(ref_path).read_bytes()
            rendered_pils.append(_png_to_pil(rendered))
            ref_pils.append(_png_to_pil(ref_bytes))
            prompt_strs.append(prompt)
            renderable_idx.append(i)
        except Exception:
            continue

    if not renderable_idx:
        return rewards

    # Batch CLIP calls
    all_images = rendered_pils + ref_pils
    img_feats  = _image_features(all_images, device)        # (2N, D)
    txt_feats  = _text_features(prompt_strs, device)        # (N, D)

    N = len(renderable_idx)
    rend_feats = img_feats[:N]    # rendered SVG features
    ref_feats  = img_feats[N:]    # reference PNG features

    clip_i = (rend_feats * ref_feats).sum(dim=1)            # (N,)
    clip_t = (rend_feats * txt_feats).sum(dim=1)            # (N,)
    clip_scores = (alpha * clip_i + (1 - alpha) * clip_t).tolist()

    for j, idx in enumerate(renderable_idx):
        svg = svgs[idx]
        bonus = _element_bonus(svg) if svg else 0.0
        rewards[idx] = float(clip_scores[j]) + bonus

    return rewards

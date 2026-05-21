"""
DiffusionSVG — Step 4: GRPO Training
======================================
MUST be run AFTER IntroSVG SFT training (../IntroSVG/train_sft.sh).
The IntroSVG M_SFT checkpoint is the starting point — it already knows how to
write clean, geometric SVG code for simple prompts. This GRPO stage extends
that ability to complex multi-object prompts using diffusion PNGs as visual targets.

Policy: ../IntroSVG/checkpoints/m_sft/epoch_3  (IntroSVG M_SFT)
Reward: α·CLIP-I(rendered_svg, ref_png) + (1-α)·CLIP-T(rendered_svg, prompt)

Algorithm: GRPO (DeepSeekMath §3.2)
  • Generate N=4 SVG candidates per prompt
  • Score each via CLIP against diffusion reference PNG
  • Advantage: A_i = (r_i − mean(r)) / (std(r) + ε)
  • Loss: −Σ A_i·log π_θ(y_i|x) + β·KL(π_θ ∥ π_ref)

Hardware: 1 × A100 40–80 GB
  (8-bit policy + frozen bf16 ref + CLIP-ViT-B/32 ≈ 28 GB total)

Run (after IntroSVG SFT completes):
    python 04_grpo_train.py \
        --model  ../IntroSVG/checkpoints/m_sft/epoch_3 \
        --data   data/grpo_train.jsonl \
        --output checkpoints/grpo_svg
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
sys.path.insert(0, str(Path(__file__).parent))
from rewards import batch_rewards

log = logging.getLogger("grpo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Defaults ──────────────────────────────────────────────────────────────────
N_SAMPLES  = 8       # candidates per prompt (GRPO group size)
BETA       = 0.04    # KL penalty coefficient
LR         = 1e-6    # learning rate
EPOCHS     = 2
MAX_NEW    = 2048    # max tokens for SVG generation
TEMP       = 0.8     # sampling temperature
EPS        = 1e-8    # numerical stability for advantage normalisation


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GRPODataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_name: str):
    from transformers import (
        Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )
    log.info(f"Loading policy model: {model_name}")
    # 8-bit for data-efficiency; use bf16 full precision if ≥ 40 GB available
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 \
              if torch.cuda.is_available() else 0
    if vram_gb >= 38:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
        )
    else:
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, quantization_config=quant, device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _make_ref_model(model_name: str):
    """Frozen reference policy — same weights, no gradient."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

_GEN_PROMPT = (
    "You are an expert SVG artist. Create a detailed SVG scene for: '{}'.\n\n"
    "STEP 1 — PLAN:\n"
    "List every object in the scene, its position in the 200×200 canvas, and its color. "
    "Example: '- Sky: blue rect (0,0)→(200,100)  - Ground: green rect (0,100)→(200,200)'\n\n"
    "STEP 2 — DRAW:\n"
    "Generate the SVG based on your plan. Rules:\n"
    "- viewBox='0 0 200 200'\n"
    "- Use <rect> for rectangular shapes (sky, ground, buildings, walls)\n"
    "- Use <circle> or ellipse paths for round shapes (sun, moon, wheels)\n"
    "- Use <path> for irregular shapes (trees, mountains, waves, animals)\n"
    "- Every element MUST have a fill= color attribute\n"
    "- Include AT LEAST 8 distinct elements covering the full canvas\n\n"
    "Write your PLAN first, then the SVG code starting with <svg."
)


def _build_input(prompt: str, processor) -> dict:
    messages = [{"role": "user",
                 "content": _GEN_PROMPT.format(prompt)}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return processor(text=[text], return_tensors="pt")


@torch.no_grad()
def _generate_group(
    prompt: str,
    policy,
    processor,
    n: int = N_SAMPLES,
    temperature: float = TEMP,
    max_new_tokens: int = MAX_NEW,
) -> Tuple[List[str], List[torch.Tensor], torch.Tensor]:
    """
    Generate n SVG candidates for one prompt.
    Returns:
      raw_texts   : list of decoded strings (length n)
      token_ids   : list of response token id tensors (variable length)
      prompt_len  : scalar — number of tokens in the prompt
    """
    inputs     = _build_input(prompt, processor).to(policy.device)
    prompt_len = inputs["input_ids"].shape[1]

    all_ids = policy.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.3,
        num_return_sequences=n,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    # all_ids: (n, prompt_len + response_len)
    response_ids = [all_ids[i, prompt_len:] for i in range(n)]
    raw_texts    = [processor.tokenizer.decode(r, skip_special_tokens=True)
                    for r in response_ids]
    return raw_texts, response_ids, prompt_len


def _extract_svg(text: str) -> Optional[str]:
    m = re.search(r'(<svg[\s>].*?</svg>)', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability computation
# ─────────────────────────────────────────────────────────────────────────────

def _seq_log_prob(
    model,
    input_ids: torch.Tensor,       # (1, prompt_len + resp_len)
    response_start: int,
) -> torch.Tensor:
    """Sum of log-probs over response tokens (scalar)."""
    out    = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]                    # (1, L-1, V)
    tgt    = input_ids[:, 1:]                          # (1, L-1)
    lp     = F.log_softmax(logits.float(), dim=-1)
    tok_lp = lp.gather(2, tgt.clamp(min=0).unsqueeze(2)).squeeze(2)  # (1, L-1)
    return tok_lp[:, response_start - 1:].sum()        # sum over response tokens


# ─────────────────────────────────────────────────────────────────────────────
# GRPO loss
# ─────────────────────────────────────────────────────────────────────────────

def _grpo_loss(
    policy,
    ref_model,
    prompt: str,
    processor,
    ref_png: str,
    n: int    = N_SAMPLES,
    beta: float = BETA,
    device: str = "cuda",
) -> Tuple[torch.Tensor, float]:
    """
    Full GRPO step for one prompt.
    Returns (loss, mean_reward).
    """
    from svg_utils import standardize_svg

    # ── Generate group ────────────────────────────────────────────────────────
    raw_texts, response_ids, prompt_len = _generate_group(
        prompt, policy, processor, n=n,
    )

    # ── Extract and standardise SVGs ─────────────────────────────────────────
    svgs = []
    for text in raw_texts:
        svg = _extract_svg(text)
        if svg:
            svg = standardize_svg(svg) or svg
        svgs.append(svg)

    # ── Compute rewards ───────────────────────────────────────────────────────
    rewards = batch_rewards(
        svgs,
        prompts=[prompt] * n,
        ref_png_paths=[ref_png] * n,
        device=device,
    )
    r = torch.tensor(rewards, dtype=torch.float32, device=device)

    # ── Group-relative advantage ──────────────────────────────────────────────
    adv = (r - r.mean()) / (r.std() + EPS)   # (n,)

    # ── Policy and reference log-probs ────────────────────────────────────────
    inputs    = _build_input(prompt, processor).to(device)
    input_ids = inputs["input_ids"]           # (1, prompt_len)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, resp_ids in enumerate(response_ids):
        full_ids = torch.cat([input_ids, resp_ids.unsqueeze(0).to(device)], dim=1)

        pi_lp  = _seq_log_prob(policy,    full_ids, prompt_len)
        with torch.no_grad():
            ref_lp = _seq_log_prob(ref_model, full_ids, prompt_len)

        kl   = pi_lp - ref_lp                        # KL per sample
        loss = -(adv[i] * pi_lp) + beta * kl
        total_loss = total_loss + loss

    return total_loss / n, float(r.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy, processor = _load_model(args.model)
    ref_model         = _make_ref_model(args.model)

    dataset = GRPODataset(args.data)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                         collate_fn=lambda x: x[0])   # one prompt at a time

    optimizer    = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    total_steps  = len(loader) * args.epochs // args.grad_accum
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step  = 0
    running_loss = 0.0
    running_rew  = 0.0

    for epoch in range(args.epochs):
        policy.train()

        for step_i, row in enumerate(loader):
            prompt  = row["prompt"]
            ref_png = row["ref_png"]

            loss, mean_rew = _grpo_loss(
                policy, ref_model, prompt, processor, ref_png,
                n=args.n_samples, beta=args.beta, device=device,
            )

            (loss / args.grad_accum).backward()

            if (step_i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step  += 1
                running_loss += loss.item()
                running_rew  += mean_rew

                if global_step % 20 == 0:
                    avg_loss = running_loss / global_step
                    avg_rew  = running_rew  / global_step
                    lr_now   = scheduler.get_last_lr()[0]
                    log.info(
                        f"Epoch {epoch+1}  step {global_step}  "
                        f"loss={avg_loss:.4f}  reward={avg_rew:.3f}  lr={lr_now:.2e}"
                    )

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = out_dir / f"epoch_{epoch+1}"
        policy.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        log.info(f"Saved → {ckpt}")

    log.info("GRPO training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="../IntroSVG/checkpoints/m_sft/epoch_3",
                        help="IntroSVG M_SFT checkpoint (must run IntroSVG SFT first)")
    parser.add_argument("--data",       default="data/grpo_train.jsonl")
    parser.add_argument("--output",     default="checkpoints/grpo_svg")
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--n-samples",  type=int,   default=N_SAMPLES,
                        help="Candidates per prompt (GRPO group size)")
    parser.add_argument("--beta",       type=float, default=BETA,
                        help="KL penalty coefficient")
    parser.add_argument("--grad-accum", type=int,   default=16)
    args = parser.parse_args()
    main(args)

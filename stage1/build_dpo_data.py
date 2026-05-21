"""
DiffuSVG — Step 3: Build DPO Preference Dataset
=================================================
Uses M_SFT to generate N candidates per prompt, scores them with GPT-4o,
and constructs (chosen, rejected) preference pairs.

Preference rules (IntroSVG §3.2):
  1. Renderable always beats non-renderable
  2. Higher GPT-4o score wins if gap ≥ δ

Prompts are drawn from the same pool used in Step 1 (PROMPT_TEMPLATES).

Output: data/d_pref_g.jsonl
    {"prompt": "...", "chosen": "<svg>...</svg>", "rejected": "<svg>...</svg>"}

Run:
    python build_dpo_data.py \
        --sft-ckpt checkpoints/m_sft/epoch_3 \
        --n-prompts 1500 \
        --n-candidates 3
"""

import argparse
import base64
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("dpo_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "d_pref_g.jsonl"

SCORE_DELTA = 1
GPT_MODEL   = "gpt-4o"
MAX_RETRIES = 5

# Import prompt pool from Step 1
from generate_sft_data import PROMPT_TEMPLATES, _make_prompt


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o scoring
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert SVG quality evaluator. "
    "Given the original text prompt and a rendered SVG image, "
    "output ONLY a single integer score from 0 to 10. No other text."
)


def _score_one(prompt: str, png_bytes: bytes, client) -> int:
    b64 = base64.b64encode(png_bytes).decode()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                        {"type": "text", "text": f'Prompt: "{prompt}"\n\nScore this SVG (0-10):'},
                    ]},
                ],
                max_tokens=4,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\d+', raw)
            return min(10, max(0, int(m.group()))) if m else 5
        except Exception as e:
            log.warning(f"GPT-4o attempt {attempt+1} failed: {e}")
            time.sleep(2.0 * (attempt + 1))
    return 5


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation using M_SFT
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(ckpt: str):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt, quantization_config=quant, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor


def _gen_candidates(prompt: str, model, processor, device, n: int) -> List[Optional[str]]:
    import torch
    from svg_utils import standardize_svg

    messages = [{"role": "user", "content": _make_prompt(prompt)}]
    text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = processor(text=[text], return_tensors="pt").to(device)

    candidates = []
    for _ in range(n):
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=1536,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        nt  = inputs["input_ids"].shape[1]
        raw = processor.tokenizer.decode(ids[0][nt:], skip_special_tokens=True)
        m   = re.search(r'(<svg[\s>].*?</svg>)', raw, re.DOTALL | re.IGNORECASE)
        svg = standardize_svg(m.group(1)) if m else None
        candidates.append(svg)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Preference pair construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_pairs(
    prompt: str,
    candidates: List[Optional[str]],
    scored: List[Tuple[int, bool]],
    delta: int,
) -> List[dict]:
    pairs = []
    n = len(candidates)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si, ri = scored[i]
            sj, rj = scored[j]
            ci, cj = candidates[i], candidates[j]
            if ci is None or cj is None:
                continue
            if ri and not rj:
                pairs.append({"prompt": prompt, "chosen": ci, "rejected": cj})
            elif ri and rj and (si - sj) >= delta:
                pairs.append({"prompt": prompt, "chosen": ci, "rejected": cj})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    import torch
    from openai import OpenAI
    from svg_utils import render_to_png, is_renderable

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample prompts from shared pool
    pool = PROMPT_TEMPLATES * ((args.n_prompts // len(PROMPT_TEMPLATES)) + 2)
    random.shuffle(pool)
    prompts = pool[:args.n_prompts]
    log.info(f"Using {len(prompts):,} prompts")

    log.info(f"Loading M_SFT from {args.sft_ckpt} ...")
    model, processor = _load_model(args.sft_ckpt)

    n_pairs = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for i, prompt in enumerate(prompts):
            candidates = _gen_candidates(prompt, model, processor, device, args.n_candidates)

            scored: List[Tuple[int, bool]] = []
            for svg in candidates:
                if svg is None:
                    scored.append((0, False))
                    continue
                png = render_to_png(svg, size=224)
                if png is None:
                    scored.append((0, False))
                else:
                    score = _score_one(prompt, png, client)
                    scored.append((score, True))

            pairs = _build_pairs(prompt, candidates, scored, delta=args.delta)
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_pairs += 1

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(prompts)}]  pairs so far: {n_pairs:,}")

    log.info(f"D_pref-G: {n_pairs:,} preference pairs → {OUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",     default="checkpoints/m_sft/epoch_3")
    parser.add_argument("--n-prompts",    type=int, default=1500)
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--delta",        type=int, default=SCORE_DELTA)
    args = parser.parse_args()
    main(args)

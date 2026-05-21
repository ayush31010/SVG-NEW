# DiffuSVG

Text-to-SVG generation using a two-stage pipeline: SFT + DPO to teach a VLM SVG syntax, then GRPO with diffusion-model visual rewards to improve complex multi-object scene quality.

**Base model**: Qwen2.5-VL-7B-Instruct (INT8 quantized, fits on 1× A100 40GB)

---

## Pipeline

```text
┌─────────────────────────── Stage 1: IntroSVG ───────────────────────────┐
│                                                                          │
│  Step 1          Step 2          Step 3           Step 4                │
│  Generate        SFT LoRA        Build DPO        DPO                   │
│  500 SVGs   →   training    →   pairs (1500×3) → training → M_Final    │
│  (base model)   rank=64          GPT-4o scored    β=0.1                 │
│                 3 epochs                           3 epochs              │
└──────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────── Stage 2: DiffuSVG ───────────────────────────┐
│                                                                          │
│  Step 5                              Step 6                             │
│  SD-Turbo reference PNGs        →   GRPO training  → Final Model       │
│  + vtracer vectorization             n=4, β=0.04                        │
│  1000 complex prompts                3 epochs                            │
│                                      reward = 0.7·CLIP-I + 0.3·CLIP-T  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
pip install -r requirements.txt

# LlamaFactory (required for Step 2 SFT):
pip install git+https://github.com/hiyouga/LLaMA-Factory.git

export OPENAI_API_KEY="sk-..."   # required for Step 3 DPO scoring
```

**Hardware**: 1× A100 40–80 GB (INT8 quantization keeps peak ~28 GB)

---

## Training

```bash
tmux new -s train
bash train.sh
# Ctrl+B D to detach; tmux attach -t train to resume
```

Each step checks if its output already exists — safe to re-run after a crash.

| Step | Script | Output | Time (A100) |
|------|--------|--------|-------------|
| 1 | `stage1/generate_sft_data.py` | `data/d_sft.jsonl` | ~8–10 h |
| 2 | LlamaFactory SFT | `checkpoints/m_sft/` | ~8–12 h |
| 3 | `stage1/build_dpo_data.py` | `data/d_pref_g.jsonl` | ~6–8 h |
| 4 | `stage1/dpo_train.py` | `checkpoints/m_final/` | ~4–6 h |
| 5 | `stage2/generate_pngs.py` + `vectorize_svgs.py` | `data/grpo_train.jsonl` | ~1 h |
| 6 | `stage2/grpo_train.py` | `checkpoints/grpo_final/` | ~8–12 h |

Total: **~38–52 hours**

---

## Inference

```bash
# Single prompt
python infer.py \
    --model stage2/checkpoints/grpo_final/epoch_3 \
    --prompt "a red barn with green fields and blue sky"

# Save to file
python infer.py \
    --model stage2/checkpoints/grpo_final/epoch_3 \
    --prompt "a lighthouse on rocky cliffs with waves below" \
    --output out.svg

# Batch from CSV (must have a 'prompt' column)
python infer.py \
    --model stage2/checkpoints/grpo_final/epoch_3 \
    --csv   prompts.csv \
    --output results/
```

---

## Repository Structure

```text
DiffuSVG/
├── train.sh                        # Full 6-step pipeline orchestration
├── infer.py                        # Inference: single prompt or CSV batch
├── requirements.txt
├── prompts.txt                     # 300 complex scene prompts (Stage 2 input)
│
├── stage1/                         # IntroSVG — SVG syntax learning
│   ├── generate_sft_data.py        # Step 1: generate 500 SVG training examples
│   ├── build_dpo_data.py           # Step 3: build GPT-4o-scored DPO pairs
│   ├── dpo_train.py                # Step 4: DPO training → M_Final
│   └── svg_utils.py                # SVG standardization, rendering, filters
│
└── stage2/                         # DiffuSVG — visual quality via GRPO
    ├── filter_prompts.py           # Select complex prompts from prompts.txt
    ├── generate_pngs.py            # Step 5a: SD-Turbo reference PNGs
    ├── vectorize_svgs.py           # Step 5b: PNG → SVG via vtracer
    ├── build_dataset.py            # Step 5c: assemble grpo_train.jsonl
    ├── grpo_train.py               # Step 6: GRPO training
    └── rewards.py                  # CLIP-I + CLIP-T reward computation
```

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| INT8 quantization (bitsandbytes) | Fits 7B policy + frozen ref on 40 GB A100 |
| Temperature sampling (T=0.9) in Step 1 | Prevents duplicate SVGs from same prompt |
| Min 3 filled elements filter | Rejects near-empty outputs without cairosvg |
| LoRA rank=64, alpha=128 | High-rank adapters needed for SVG token syntax |
| n=4 candidates in GRPO | Group size balances diversity vs. GPU memory |
| α=0.7 CLIP-I + 0.3 CLIP-T | Visual fidelity weighted higher than text alignment |
| SD-Turbo reference PNGs | 1-step diffusion provides fast high-quality targets |
| Layered scene prompting | Background → midground → foreground improves coherence |

---

## References

- **IntroSVG** (CVPR 2026): SFT + DPO training of VLMs for SVG generation
- **Qwen2.5-VL**: [github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- **LlamaFactory**: [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **GRPO**: DeepSeekMath §3.2 — group relative policy optimization

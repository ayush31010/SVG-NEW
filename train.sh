#!/usr/bin/env bash
# DiffuSVG — Full Training Pipeline
# ===================================
# Stage 1 (IntroSVG): Teach Qwen2.5-VL-7B to generate SVGs via SFT + DPO
# Stage 2 (DiffuSVG): Boost complex-scene quality via GRPO + SD-Turbo rewards
#
# Time estimate on 1× A100 40–80 GB:
#   Step 1  Generate SFT data     500 samples       ~8–10 hrs
#   Step 2  SFT LoRA (rank=64)    3 epochs          ~8–12 hrs
#   Step 3  Build DPO pairs       1500×3 candidates ~6–8 hrs
#   Step 4  DPO training          3 epochs          ~4–6 hrs
#   Step 5  Reference PNGs + vectorize              ~1 hr
#   Step 6  GRPO training         3 epochs          ~8–12 hrs
#   Model downloads (first run)                     ~2 hrs
#   ────────────────────────────────────────────────────────
#   Total                                           ~38–52 hrs
#
# Resumable: each step skips if its output already exists.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."   # required for Step 3 DPO scoring
#   tmux new -s train
#   bash train.sh
#   # Ctrl+B D to detach;  tmux attach -t train to resume

set -euo pipefail

LOG="train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%H:%M:%S'; }

skip_if_exists() {
    local path="$1" label="$2"
    if [ -d "$path" ] || { [ -f "$path" ] && [ -s "$path" ]; }; then
        echo "  [SKIP] $label already exists"
        return 0
    fi
    return 1
}

echo "========================================================"
echo "  DiffuSVG Training Pipeline"
echo "  Started: $(date)"
echo "========================================================"

# ── STAGE 1: IntroSVG ─────────────────────────────────────────────────────────

cd stage1
mkdir -p data checkpoints

echo ""
echo "════════════════════════════════════════════════════════"
echo "  STAGE 1 — IntroSVG (SVG Syntax Learning)"
echo "════════════════════════════════════════════════════════"

# Step 1: Generate SFT data
echo ""
echo "[1/6] Generating SFT dataset (500 SVG examples)..."
echo "  Expected: ~8–10 hrs on A100"
echo "  ⏱ $(ts)"
skip_if_exists data/d_sft.jsonl "data/d_sft.jsonl" || \
PYTHONUNBUFFERED=1 python generate_sft_data.py \
    --n-samples 1000
echo "  ⏱ $(ts)"

# Step 2: SFT LoRA training via LlamaFactory
echo ""
echo "[2/6] SFT training (LoRA rank=64, 3 epochs)..."
echo "  Expected: ~8–12 hrs on A100"
echo "  ⏱ $(ts)"
skip_if_exists checkpoints/m_sft/epoch_3 "checkpoints/m_sft/epoch_3" || \
llamafactory-cli train \
    --model_name_or_path             Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset                        d_sft \
    --dataset_dir                    ./data \
    --template                       qwen2_vl \
    --stage                          sft \
    --finetuning_type                lora \
    --lora_rank                      64 \
    --lora_alpha                     128 \
    --lora_target                    all \
    --cutoff_len                     4096 \
    --per_device_train_batch_size    1 \
    --gradient_accumulation_steps    128 \
    --lr_scheduler_type              cosine \
    --warmup_ratio                   0.03 \
    --learning_rate                  5e-5 \
    --num_train_epochs               3.0 \
    --bf16                           true \
    --output_dir                     checkpoints/m_sft \
    --save_steps                     500 \
    --logging_steps                  50 \
    --report_to                      none
echo "  ⏱ $(ts)"

# Step 3: Build DPO preference dataset
echo ""
echo "[3/6] Building DPO preference dataset (1500 prompts × 3 candidates)..."
echo "  Expected: ~6–8 hrs on A100 + OpenAI API cost"
echo "  ⏱ $(ts)"
skip_if_exists data/d_pref_g.jsonl "data/d_pref_g.jsonl" || \
PYTHONUNBUFFERED=1 python build_dpo_data.py \
    --sft-ckpt     checkpoints/m_sft/epoch_3 \
    --n-prompts    1500 \
    --n-candidates 5 \
    --delta        1
echo "  ⏱ $(ts)"

# Step 4: DPO training → M_Final
echo ""
echo "[4/6] DPO training (3 epochs)..."
echo "  Expected: ~4–6 hrs on A100"
echo "  ⏱ $(ts)"
skip_if_exists checkpoints/m_final/epoch_3 "checkpoints/m_final/epoch_3" || \
python dpo_train.py \
    --sft-ckpt         checkpoints/m_sft/epoch_3 \
    --data             data/d_pref_g.jsonl \
    --output           checkpoints/m_final \
    --epochs           3 \
    --per-device-batch 1 \
    --grad-accum       16 \
    --lr               5e-6 \
    --beta             0.1
echo "  ⏱ $(ts)"

cd ..

# ── STAGE 2: DiffuSVG ─────────────────────────────────────────────────────────

cd stage2
mkdir -p data checkpoints

echo ""
echo "════════════════════════════════════════════════════════"
echo "  STAGE 2 — DiffuSVG (Visual Quality via GRPO)"
echo "════════════════════════════════════════════════════════"

# Step 5: Filter complex prompts → SD-Turbo PNGs → vectorize → build dataset
echo ""
echo "[5/6] Generating reference PNGs and vectorizing to SVG..."
echo "  Expected: ~1 hr on A100"
echo "  ⏱ $(ts)"

skip_if_exists data/complex_prompts.jsonl "complex_prompts.jsonl" || \
python filter_prompts.py \
    --input     ../prompts.txt \
    --output    data/complex_prompts.jsonl \
    --min-score 2

# Cap to 1000 complex prompts
python -c "
lines = open('data/complex_prompts.jsonl').readlines()
open('data/complex_prompts.jsonl','w').writelines(lines[:1000])
print(f'Capped to {min(len(lines),1000)} complex prompts')
"

skip_if_exists data/complex_prompts_with_ids.jsonl "ref PNGs" || \
PYTHONUNBUFFERED=1 python generate_pngs.py \
    --input  data/complex_prompts.jsonl \
    --output data/ref_pngs/

skip_if_exists data/vectorized.jsonl "vectorized SVGs" || \
python vectorize_svgs.py \
    --input   data/complex_prompts_with_ids.jsonl \
    --png-dir data/ref_pngs/ \
    --svg-dir data/ref_svgs/ \
    --workers 8

python build_dataset.py
echo "  ⏱ $(ts)"

# Step 6: GRPO training → Final model
echo ""
echo "[6/6] GRPO training (3 epochs from M_Final)..."
echo "  Expected: ~8–12 hrs on A100"
echo "  ⏱ $(ts)"
skip_if_exists checkpoints/grpo_final/epoch_3 "checkpoints/grpo_final/epoch_3" || \
PYTHONUNBUFFERED=1 python grpo_train.py \
    --model      "../stage1/checkpoints/m_final/epoch_3" \
    --data       data/grpo_train.jsonl \
    --output     checkpoints/grpo_final \
    --epochs     3 \
    --n-samples  8 \
    --beta       0.04 \
    --grad-accum 16
echo "  ⏱ $(ts)"

cd ..

echo ""
echo "========================================================"
echo "  TRAINING COMPLETE"
echo "  Finished: $(date)"
echo "  Final model: stage2/checkpoints/grpo_final/epoch_3"
echo ""
echo "  Inference:"
echo "    python infer.py \\"
echo "        --model stage2/checkpoints/grpo_final/epoch_3 \\"
echo "        --prompt 'a red barn with green fields and blue sky'"
echo "========================================================"

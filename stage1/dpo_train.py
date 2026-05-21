"""
IntroSVG — Step 5: DPO Training  → M_Final
===========================================
Paper §4.2:
  • Policy model  M_θ   : initialised from M_SFT
  • Reference model M_ref: M_SFT (frozen)
  • Dataset        : D_pref-G (generation prompts only)
  • Loss           : standard DPO loss, β = 0.1
  • lr = 5e-6, cosine decay, 3 epochs, 8 × A100/A800 80 GB

Run (single node, 8 GPUs):
    accelerate launch \
        --config_file deepspeed_zero3.json \
        --num_processes 1 \
        05_dpo_train.py

Run (single GPU, for testing):
    python 05_dpo_train.py --per-device-batch 1 --grad-accum 16
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

log = logging.getLogger("step5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR   = Path("data")
D_PREF     = DATA_DIR / "d_pref_g.jsonl"
SFT_CKPT   = "checkpoints/m_sft/epoch_3"
OUT_DIR    = Path("checkpoints/m_final")

LR      = 5e-6
BETA    = 0.1       # KL-divergence penalty coefficient
EPOCHS  = 3
MAX_LEN = 3072


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DPODataset(torch.utils.data.Dataset):
    """
    Each row: {"prompt": str, "chosen": svg_str, "rejected": svg_str}
    Encodes prompt+chosen and prompt+rejected into input_ids tensors,
    with labels masked on the prompt portion (only response is supervised).
    """

    def __init__(self, jsonl_path: str, processor, max_len: int = MAX_LEN):
        self.rows: List[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.processor = processor
        self.max_len   = max_len

    def __len__(self):
        return len(self.rows)

    def _encode_pair(
        self, prompt_text: str, response_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, labels) for one (prompt, response) pair."""
        prompt_ids = self.processor.tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        resp_ids = self.processor.tokenizer.encode(
            response_text + self.processor.tokenizer.eos_token,
            add_special_tokens=False,
        )
        input_ids = torch.tensor(
            prompt_ids + resp_ids, dtype=torch.long
        )[:self.max_len]
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100   # mask prompt
        return input_ids, labels

    def __getitem__(self, idx: int) -> dict:
        row    = self.rows[idx]
        prompt = row["prompt"]

        from svg_utils import gen_prompt
        prompt_text = self.processor.apply_chat_template(
            [{"role": "user", "content": gen_prompt(prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )

        ch_ids, ch_lab = self._encode_pair(prompt_text, row["chosen"])
        rj_ids, rj_lab = self._encode_pair(prompt_text, row["rejected"])

        return {
            "chosen_input_ids":      ch_ids,
            "chosen_labels":         ch_lab,
            "chosen_attention_mask": torch.ones_like(ch_ids),
            "rejected_input_ids":    rj_ids,
            "rejected_labels":       rj_lab,
            "rejected_attention_mask": torch.ones_like(rj_ids),
        }


def _pad(tensors: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_val, dtype=torch.long)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _collate_dpo(batch: List[dict], pad_id: int) -> dict:
    keys = ["chosen_input_ids", "chosen_labels", "chosen_attention_mask",
            "rejected_input_ids", "rejected_labels", "rejected_attention_mask"]
    out = {}
    for k in keys:
        pad = -100 if "label" in k else pad_id
        out[k] = _pad([b[k] for b in batch], pad)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────────────────────

def _log_probs(model, input_ids, labels, attention_mask) -> torch.Tensor:
    """Compute per-token log-probs for response tokens only."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]          # shift left
        tgt    = labels[:, 1:]                   # shift right

    log_p   = torch.log_softmax(logits.float(), dim=-1)
    tok_lp  = log_p.gather(2, tgt.clamp(min=0).unsqueeze(2)).squeeze(2)
    mask    = (tgt != -100).float()
    return (tok_lp * mask).sum(dim=1)            # sum over response tokens


def _dpo_loss(
    policy_model,
    ref_model,
    batch: dict,
    beta: float,
) -> torch.Tensor:
    """
    Standard DPO loss (Rafailov et al., 2023).
    L_DPO = -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x)))]
    """
    ch_ids  = batch["chosen_input_ids"]
    ch_lab  = batch["chosen_labels"]
    ch_mask = batch["chosen_attention_mask"]
    rj_ids  = batch["rejected_input_ids"]
    rj_lab  = batch["rejected_labels"]
    rj_mask = batch["rejected_attention_mask"]

    pi_ch = _log_probs(policy_model, ch_ids, ch_lab, ch_mask)
    pi_rj = _log_probs(policy_model, rj_ids, rj_lab, rj_mask)

    with torch.no_grad():
        ref_ch = _log_probs(ref_model, ch_ids, ch_lab, ch_mask)
        ref_rj = _log_probs(ref_model, rj_ids, rj_lab, rj_mask)

    logits_w = pi_ch - ref_ch
    logits_l = pi_rj - ref_rj
    loss = -torch.nn.functional.logsigmoid(beta * (logits_w - logits_l)).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        get_cosine_schedule_with_warmup,
    )
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    device   = accelerator.device
    is_main  = accelerator.is_main_process

    if is_main:
        log.info(f"DPO training  |  {accelerator.num_processes} GPU(s)")
        log.info(f"  SFT ckpt : {args.sft_ckpt}")
        log.info(f"  β = {args.beta}  |  lr = {args.lr}  |  epochs = {args.epochs}")

    processor = AutoProcessor.from_pretrained(args.sft_ckpt)

    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    # Policy model — 8-bit base + LoRA adapters only trained.
    # Gradient memory = LoRA params (~400 MB) not full 7B, fits 40 GB A100.
    policy_quant = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["visual"],
    )
    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.sft_ckpt,
        quantization_config=policy_quant,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.gradient_checkpointing_enable()

    # Reference model — 8-bit, frozen (forward pass only, ~7 GB)
    ref_quant = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["visual"],
    )
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.sft_ckpt,
        quantization_config=ref_quant,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Dataset + DataLoader
    pad_id  = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    dataset = DPODataset(args.data, processor, max_len=args.max_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: _collate_dpo(b, pad_id),
    )

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(loader) * args.epochs // args.grad_accum
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    policy, optimizer, loader, scheduler = accelerator.prepare(
        policy, optimizer, loader, scheduler
    )
    ref_model = ref_model.to(device)

    global_step = 0
    for epoch in range(args.epochs):
        policy.train()
        running_loss = 0.0

        for batch in loader:
            with accelerator.accumulate(policy):
                loss = _dpo_loss(policy, ref_model, batch, beta=args.beta)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step   += 1
                running_loss  += loss.item()
                if is_main and global_step % 50 == 0:
                    avg = running_loss / global_step
                    lr  = scheduler.get_last_lr()[0]
                    log.info(f"Epoch {epoch+1}  step {global_step}  dpo_loss={avg:.4f}  lr={lr:.2e}")

        if is_main:
            ckpt = Path(args.output) / f"epoch_{epoch+1}"
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(policy)
            unwrapped.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            log.info(f"Saved → {ckpt}")

    if is_main:
        log.info("DPO training complete → M_Final saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",        default=SFT_CKPT)
    parser.add_argument("--data",            default=str(D_PREF))
    parser.add_argument("--output",          default=str(OUT_DIR))
    parser.add_argument("--lr",              type=float, default=LR)
    parser.add_argument("--beta",            type=float, default=BETA)
    parser.add_argument("--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--per-device-batch",type=int,   default=1)
    parser.add_argument("--grad-accum",      type=int,   default=16)
    parser.add_argument("--max-len",         type=int,   default=MAX_LEN)
    args = parser.parse_args()
    main(args)

"""
DiffuSVG Inference
==================
Generate SVG from a text prompt using the trained model.

Usage:
    python infer.py --model stage2/checkpoints/grpo_final/epoch_3 \
                    --prompt "a red barn with green fields and blue sky"

    python infer.py --model stage2/checkpoints/grpo_final/epoch_3 \
                    --csv prompts.csv --output results/
"""

import argparse
import re
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "stage1"))


def load_model(model_path: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=quant, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def generate_svg(prompt: str, model, processor, max_new_tokens: int = 1536) -> str:
    instruction = (
        f"You are an expert SVG artist. Create a detailed SVG scene for: '{prompt}'.\n\n"
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
    messages = [{"role": "user", "content": instruction}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(torch.device("cuda"))

    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    n = inputs["input_ids"].shape[1]
    raw = processor.tokenizer.decode(ids[0][n:], skip_special_tokens=True).strip()
    m = re.search(r"(<svg[\s>].*?</svg>)", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else raw


def main(args):
    print(f"Loading model from {args.model}...")
    model, processor = load_model(args.model)

    if args.prompt:
        svg = generate_svg(args.prompt, model, processor)
        if args.output:
            Path(args.output).write_text(svg, encoding="utf-8")
            print(f"Saved → {args.output}")
        else:
            print(svg)
        return

    if args.csv:
        import csv
        out_dir = Path(args.output or "results")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                prompt = row.get("prompt", row.get("caption", ""))
                if not prompt:
                    continue
                svg = generate_svg(prompt, model, processor)
                out_path = out_dir / f"{i:04d}.svg"
                out_path.write_text(svg, encoding="utf-8")
                print(f"[{i+1}] {prompt[:60]}... → {out_path}")
        return

    print("Provide --prompt or --csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to trained model checkpoint")
    parser.add_argument("--prompt", default="",    help="Single text prompt")
    parser.add_argument("--csv",    default="",    help="CSV file with 'prompt' column")
    parser.add_argument("--output", default="",    help="Output file or directory")
    main(parser.parse_args())

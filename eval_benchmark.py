"""
eval_benchmark.py — LLM SVG Generation Benchmark Evaluation
=============================================================
Runs all 30 prompts from Simon Willison / Tom Gally's LLM SVG Generation
Benchmark through the trained DiffuSVG model and produces:

  1. SVG files       → eval_out/svgs/<n>.svg
  2. PNG renders     → eval_out/pngs/<n>.png
  3. CLIP-T scores   → eval_out/scores.json
  4. HTML gallery    → eval_out/index.html  (open in browser to compare)

Usage (after full pipeline completes):
    python eval_benchmark.py \
        --model stage2/checkpoints/grpo_final/epoch_3

Or compare against an earlier checkpoint:
    python eval_benchmark.py \
        --model stage1/checkpoints/m_final/epoch_3 \
        --tag dpo
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ── 30 benchmark prompts ──────────────────────────────────────────────────────
BENCHMARK_PROMPTS = [
    "an octopus operating a pipe organ",
    "a giraffe assembling a grandfather clock",
    "a starfish driving a bulldozer",
    "a moose conducting a carousel",
    "a flamingo repairing a telescope",
    "a hedgehog playing an accordion",
    "a jellyfish piloting a Ferris wheel",
    "an elephant typing on a typewriter",
    "a chameleon tuning a grand piano",
    "a penguin juggling chainsaws",
    "a sloth steering an excavator",
    "a dragonfly balancing a chandelier",
    "a rhinoceros painting a lighthouse",
    "a seahorse examining a microscope",
    "a peacock spinning a pottery wheel",
    "a kangaroo climbing a radio tower",
    "a lobster polishing a harp",
    "a porcupine pushing a lawnmower",
    "a gecko installing a satellite dish",
    "an iguana carving a totem pole",
    "an armadillo lifting a drawbridge",
    "a mantis studying a sextant",
    "an ostrich pulling a rickshaw",
    "a squid disassembling a printing press",
    "a butterfly inspecting a steam engine",
    "a crab descending a fire escape",
    "a venus flytrap swallowing a street lamp",
    "coral cleaning a ship's wheel",
    "a sea anemone threading a loom",
    "an orchid supporting a pergola",
]

GEN_PROMPT = (
    "You are an expert SVG artist. Create a detailed SVG scene for: '{}'.\n\n"
    "STEP 1 — PLAN:\n"
    "List every object in the scene, its position in the 200×200 canvas, and its color.\n"
    "Example: '- Background: blue rect (0,0)→(200,200)  - Body: green ellipse at (100,120)'\n\n"
    "STEP 2 — DRAW:\n"
    "Generate the SVG based on your plan. Rules:\n"
    "- viewBox='0 0 200 200'\n"
    "- Use <rect> for rectangular shapes\n"
    "- Use <circle> or <ellipse> for round shapes\n"
    "- Use <path> for irregular shapes\n"
    "- Every element MUST have a fill= color attribute\n"
    "- Include AT LEAST 8 distinct elements covering the full canvas\n\n"
    "Write your PLAN first, then the SVG code starting with <svg."
)


def _load_model(model_path: str):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"GPU VRAM: {vram:.1f} GB")

    if vram >= 38:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
        )
    else:
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quant, device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def _generate_svg(prompt: str, model, processor, device: str) -> str | None:
    import torch
    msg = [{"role": "user", "content": GEN_PROMPT.format(prompt)}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.3,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    n = inputs["input_ids"].shape[1]
    raw = processor.tokenizer.decode(ids[0][n:], skip_special_tokens=True).strip()
    m = re.search(r'(<svg[\s>].*?</svg>)', raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _render_png(svg: str, size: int = 400) -> bytes | None:
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    except Exception:
        return None


def _clip_t_score(png_bytes: bytes, prompt: str, device: str) -> float:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        import io

        if not hasattr(_clip_t_score, "_model"):
            _clip_t_score._model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(device).eval()
            _clip_t_score._proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        model = _clip_t_score._model
        proc  = _clip_t_score._proc

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        inputs = proc(text=[prompt], images=[img], return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(device)
        with torch.no_grad():
            out = model(**inputs)
            score = (out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)) @ \
                    (out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)).T
        return float(score[0, 0])
    except Exception:
        return 0.0


def _element_count(svg: str) -> int:
    return len(re.findall(r'<(?:path|rect|circle|ellipse|polygon|polyline)[^>]+fill', svg))


def _make_html(results: list, out_dir: Path, model_tag: str) -> None:
    rows = ""
    for r in results:
        idx      = r["index"]
        prompt   = r["prompt"]
        score    = r["clip_t"]
        nelems   = r["elements"]
        rendered = r["rendered"]
        png_rel  = f"pngs/{idx:02d}.png" if rendered else ""

        img_html = (f'<img src="{png_rel}" width="300" height="300" '
                    f'style="border:1px solid #ccc;border-radius:6px;">'
                    if rendered else
                    '<div style="width:300px;height:300px;background:#fee2e2;'
                    'display:flex;align-items:center;justify-content:center;'
                    'border-radius:6px;color:#991b1b;font-size:13px;">'
                    'Failed to render</div>')

        score_color = "#065f46" if score >= 0.28 else "#92400e" if score >= 0.22 else "#991b1b"

        rows += f"""
        <tr>
          <td style="padding:8px;text-align:center;font-weight:bold;color:#6b7280">{idx+1}</td>
          <td style="padding:8px;max-width:220px">{prompt}</td>
          <td style="padding:8px;text-align:center">{img_html}</td>
          <td style="padding:8px;text-align:center;font-size:18px;font-weight:bold;
              color:{score_color}">{score:.3f}</td>
          <td style="padding:8px;text-align:center">{nelems}</td>
        </tr>"""

    scores = [r["clip_t"] for r in results if r["rendered"]]
    avg    = sum(scores) / len(scores) if scores else 0
    best   = max(results, key=lambda r: r["clip_t"])
    worst  = min(results, key=lambda r: r["clip_t"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DiffuSVG — Benchmark Results ({model_tag})</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px;
         background: #f8fafc; color: #1e293b; }}
  h1   {{ color: #1e40af; margin-bottom: 4px; }}
  .sub {{ color: #64748b; margin-bottom: 24px; font-size: 14px; }}
  .stat-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px;
           padding: 16px 24px; min-width: 140px; }}
  .stat-val {{ font-size: 28px; font-weight: 700; color: #1e40af; }}
  .stat-lbl {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 12px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th    {{ background: #1e40af; color: white; padding: 12px 8px;
           font-size: 13px; text-align: center; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  tr:hover {{ background: #eff6ff; }}
  .bench-link {{ margin-bottom: 16px; font-size: 13px; }}
  .bench-link a {{ color: #2563eb; }}
</style>
</head>
<body>

<h1>DiffuSVG — LLM SVG Generation Benchmark</h1>
<p class="sub">Model: <code>{model_tag}</code> &nbsp;·&nbsp;
   30 prompts from <a href="https://simonwillison.net/2025/Nov/25/llm-svg-generation-benchmark/"
   target="_blank">Simon Willison / Tom Gally benchmark</a></p>

<div class="stat-row">
  <div class="stat">
    <div class="stat-val">{avg:.3f}</div>
    <div class="stat-lbl">Avg CLIP-T</div>
  </div>
  <div class="stat">
    <div class="stat-val">{sum(1 for r in results if r["rendered"])}/{len(results)}</div>
    <div class="stat-lbl">Rendered OK</div>
  </div>
  <div class="stat">
    <div class="stat-val">{sum(r["elements"] for r in results)//len(results)}</div>
    <div class="stat-lbl">Avg Elements</div>
  </div>
  <div class="stat">
    <div class="stat-val" style="font-size:14px;padding-top:6px">
      {best["prompt"][:30]}…</div>
    <div class="stat-lbl">Best prompt (CLIP-T {best["clip_t"]:.3f})</div>
  </div>
  <div class="stat">
    <div class="stat-val" style="font-size:14px;padding-top:6px">
      {worst["prompt"][:30]}…</div>
    <div class="stat-lbl">Worst prompt (CLIP-T {worst["clip_t"]:.3f})</div>
  </div>
</div>

<p class="bench-link">
  Compare visually:
  <a href="https://gally.net/temp/20251107pelican-alternatives/index.html" target="_blank">
    gally.net benchmark gallery (GPT-4o, Claude, Gemini outputs)
  </a>
</p>

<table>
<thead>
  <tr>
    <th>#</th>
    <th>Prompt</th>
    <th>Generated SVG (400×400)</th>
    <th>CLIP-T</th>
    <th>Elements</th>
  </tr>
</thead>
<tbody>{rows}
</tbody>
</table>

</body>
</html>"""

    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\nHTML gallery → {out_dir / 'index.html'}")


def main(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    (out_dir / "svgs").mkdir(parents=True, exist_ok=True)
    (out_dir / "pngs").mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, processor = _load_model(args.model)

    results = []
    scores_path = out_dir / "scores.json"

    # Resume: load existing results
    if scores_path.exists():
        results = json.loads(scores_path.read_text())
        done_idx = {r["index"] for r in results}
        print(f"Resuming — {len(done_idx)} already done")
    else:
        done_idx = set()

    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        if i in done_idx:
            continue

        print(f"\n[{i+1:02d}/30] {prompt}")

        svg = _generate_svg(prompt, model, processor, device)
        if svg is None:
            print("  ✗ no SVG extracted")
            results.append({"index": i, "prompt": prompt, "rendered": False,
                            "clip_t": 0.0, "elements": 0})
            scores_path.write_text(json.dumps(results, indent=2))
            continue

        (out_dir / "svgs" / f"{i:02d}.svg").write_text(svg, encoding="utf-8")

        png = _render_png(svg, size=400)
        rendered = png is not None
        if rendered:
            (out_dir / "pngs" / f"{i:02d}.png").write_bytes(png)
            score = _clip_t_score(png, prompt, device)
            nelems = _element_count(svg)
            print(f"  ✓ rendered  CLIP-T={score:.3f}  elements={nelems}")
        else:
            score = 0.0
            nelems = _element_count(svg)
            print(f"  ✗ render failed  elements={nelems}")

        results.append({"index": i, "prompt": prompt, "rendered": rendered,
                        "clip_t": score, "elements": nelems})
        scores_path.write_text(json.dumps(results, indent=2))

    # Sort by index for display
    results.sort(key=lambda r: r["index"])

    # Summary
    rendered_scores = [r["clip_t"] for r in results if r["rendered"]]
    print("\n" + "="*50)
    print(f"  Prompts:       30")
    print(f"  Rendered OK:   {sum(1 for r in results if r['rendered'])}/30")
    print(f"  Avg CLIP-T:    {sum(rendered_scores)/len(rendered_scores):.4f}" if rendered_scores else "  No renders")
    print(f"  Max CLIP-T:    {max(rendered_scores):.4f}" if rendered_scores else "")
    print(f"  Min CLIP-T:    {min(rendered_scores):.4f}" if rendered_scores else "")
    print("="*50)

    _make_html(results, out_dir, tag := Path(args.model).parent.name + "/" + Path(args.model).name)
    print(f"\nDone. Open {out_dir}/index.html in a browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (e.g. stage2/checkpoints/grpo_final/epoch_3)")
    parser.add_argument("--out", default="eval_out",
                        help="Output directory for SVGs, PNGs, HTML gallery")
    args = parser.parse_args()
    main(args)

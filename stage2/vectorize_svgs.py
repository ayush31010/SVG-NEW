"""
DiffusionSVG — Step 2: Raster PNG → SVG via Vectorization
===========================================================
Converts each reference PNG to SVG using vtracer (colour-accurate spline tracing).
The resulting SVG is then passed through IntroSVG's standardise_svg() to normalise
it to the same format the VLM was trained to produce (viewBox 0 0 200 200, M/L/C/A/Z).

This SVG becomes the "gold" target in the critic/correction dataset, and
the visual reference for the GRPO reward (via CLIP-I comparison).

Output:
  data/ref_svgs/{id:06d}.svg
  data/vectorized.jsonl   — extends complex_prompts_with_ids.jsonl with "ref_svg" field

Run:
    python 02_vectorize_svgs.py \
        --input  data/complex_prompts_with_ids.jsonl \
        --png-dir data/ref_pngs/ \
        --svg-dir data/ref_svgs/
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("step2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add IntroSVG to path for svg_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))


# ─────────────────────────────────────────────────────────────────────────────
# Vectorization
# ─────────────────────────────────────────────────────────────────────────────

def _vectorize_png_bytes(png_bytes: bytes) -> str:
    """Convert PNG bytes → raw SVG string via vtracer."""
    import vtracer
    return vtracer.convert_raw_image_to_svg(
        png_bytes,
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=4,       # remove noise blobs smaller than 4px
        color_precision=6,      # colour quantization levels
        layer_difference=16,    # min brightness diff between layers
        corner_threshold=60,    # angle threshold for corners (degrees)
        length_threshold=4.0,   # minimum path segment length
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,       # decimal places (will be rounded by standardise)
    )


def _process_one(args_tuple) -> dict:
    """Worker: vectorize one PNG and standardise the SVG. Returns updated row."""
    row, png_path, svg_path = args_tuple
    from svg_utils import standardize_svg, is_renderable

    if Path(svg_path).exists():
        row["ref_svg"] = svg_path
        return row

    try:
        png_bytes = Path(png_path).read_bytes()
        raw_svg   = _vectorize_png_bytes(png_bytes)
    except Exception as e:
        row["ref_svg"] = None
        row["vectorize_error"] = str(e)
        return row

    std_svg = standardize_svg(raw_svg)
    if std_svg is None or not is_renderable(std_svg):
        # Fall back to raw SVG if standardisation fails
        std_svg = raw_svg

    Path(svg_path).write_text(std_svg, encoding="utf-8")
    row["ref_svg"] = svg_path
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    svg_dir = Path(args.svg_dir)
    svg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows):,} rows from {args.input}")

    work = []
    for row in rows:
        sample_id = row.get("id", rows.index(row))
        png_path  = row.get("ref_png", str(Path(args.png_dir) / f"{sample_id:06d}.png"))
        svg_path  = str(svg_dir / f"{sample_id:06d}.svg")
        work.append((row, png_path, svg_path))

    out_file = Path("data/vectorized.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_file, "w", encoding="utf-8") as fout, \
         ProcessPoolExecutor(max_workers=args.workers) as pool:

        futures = {pool.submit(_process_one, w): w for w in work}
        for fut in as_completed(futures):
            result = fut.result()
            if result.get("ref_svg"):
                n_ok += 1
            else:
                n_fail += 1
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (n_ok + n_fail) % 100 == 0:
                log.info(f"  {n_ok+n_fail}/{len(work)}  ok={n_ok}  fail={n_fail}")

    log.info(f"Done. ok={n_ok:,}  failed={n_fail:,} → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="data/complex_prompts_with_ids.jsonl")
    parser.add_argument("--png-dir", default="data/ref_pngs")
    parser.add_argument("--svg-dir", default="data/ref_svgs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args)

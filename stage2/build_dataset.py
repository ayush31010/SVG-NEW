"""
DiffusionSVG — Step 3: Build GRPO Training Dataset
====================================================
Merges vectorized.jsonl into grpo_train.jsonl for 04_grpo_train.py.

NOTE: No SFT warmup on vtracer SVGs is generated here.
  The IntroSVG M_SFT checkpoint already provides clean SVG generation ability.
  SFT-ing on vtracer outputs would overwrite that with low-quality traced paths.
  GRPO directly on top of M_SFT is the correct approach.

Output:
  data/grpo_train.jsonl
      Each row:
      {
        "prompt":     "a red house with green windows and a sun above it",
        "ref_png":    "data/ref_pngs/000042.png",   # diffusion reference PNG
        "ref_svg":    "data/ref_svgs/000042.svg",   # vectorized reference SVG
        "complexity": 4
      }

Run:
    python 03_build_dataset.py
"""

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("step3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))

DATA_DIR  = Path("data")
IN_FILE   = DATA_DIR / "vectorized.jsonl"
GRPO_FILE = DATA_DIR / "grpo_train.jsonl"


def main():
    rows = []
    with open(IN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows):,} vectorized rows")

    n_ok = n_skip = 0

    with open(GRPO_FILE, "w", encoding="utf-8") as fg:
        for row in rows:
            ref_png = row.get("ref_png")
            ref_svg = row.get("ref_svg")
            prompt  = row.get("prompt", "")

            if not ref_png or not ref_svg or not prompt:
                n_skip += 1
                continue
            if not Path(ref_png).exists() or not Path(ref_svg).exists():
                n_skip += 1
                continue

            grpo_row = {
                "prompt":     prompt,
                "ref_png":    ref_png,
                "ref_svg":    ref_svg,
                "complexity": row.get("complexity_score", 1),
            }
            fg.write(json.dumps(grpo_row, ensure_ascii=False) + "\n")
            n_ok += 1

    log.info(f"GRPO dataset: {n_ok:,} rows → {GRPO_FILE}")
    log.info(f"Skipped     : {n_skip:,} (missing files)")
    log.info("Next: run 04_grpo_train.py --model ../IntroSVG/checkpoints/m_sft/epoch_3")


if __name__ == "__main__":
    main()

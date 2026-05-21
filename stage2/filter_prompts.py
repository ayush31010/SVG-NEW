"""
DiffusionSVG — Step 0: Filter Complex Prompts
===============================================
Only complex prompts go through the diffusion → vectorize → GRPO pipeline.
Simple prompts (single object, short, no spatial relations) are skipped.

Complexity heuristics (any one triggers "complex"):
  • ≥ 3 distinct noun phrases
  • contains a spatial/relational preposition (above, beside, next to, …)
  • ≥ 10 words
  • contains a colour modifier AND a shape AND another object

Input:  prompts.txt  (one prompt per line)  OR  a .jsonl with a "prompt" key
Output: data/complex_prompts.jsonl   {"prompt": "...", "complexity_score": N}

Run:
    python 00_filter_prompts.py --input prompts.txt --output data/complex_prompts.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("step0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Complexity signals ────────────────────────────────────────────────────────

_SPATIAL = re.compile(
    r'\b(above|below|beside|next to|in front of|behind|between|on top of|'
    r'underneath|overlapping|surrounding|inside|outside|near|adjacent|'
    r'left of|right of|above the|below the)\b',
    re.IGNORECASE,
)

_CONJUNCTIONS = re.compile(r'\b(and|with|plus|alongside|together with)\b', re.IGNORECASE)

_COLORS = re.compile(
    r'\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey|'
    r'golden|silver|cyan|magenta|violet|indigo|teal|crimson|scarlet|azure)\b',
    re.IGNORECASE,
)

_MIN_WORDS_FOR_COMPLEX = 10
_MIN_NOUN_PHRASES      = 3


def _count_noun_phrases(text: str) -> int:
    """Lightweight noun-phrase counter using spaCy if available, else word heuristic."""
    try:
        import spacy
        if not hasattr(_count_noun_phrases, "_nlp"):
            _count_noun_phrases._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            _count_noun_phrases._nlp.enable_pipe("senter")
        doc = _count_noun_phrases._nlp(text)
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
        return len(nouns)
    except Exception:
        # Fallback: count capitalised or content words
        words = text.split()
        return sum(1 for w in words if w[0].isalpha() and w.lower() not in
                   {"a", "an", "the", "of", "in", "on", "with", "and", "or", "is", "are"})


def complexity_score(prompt: str) -> Tuple[int, List[str]]:
    """
    Return (score, reasons).  Score ≥ 1 → complex.
    """
    score = 0
    reasons: List[str] = []
    words = prompt.split()

    if len(words) >= _MIN_WORDS_FOR_COMPLEX:
        score += 1
        reasons.append(f"long ({len(words)} words)")

    if _SPATIAL.search(prompt):
        score += 2
        reasons.append("spatial relation")

    n_conj = len(_CONJUNCTIONS.findall(prompt))
    if n_conj >= 1:
        score += n_conj
        reasons.append(f"{n_conj} conjunction(s)")

    n_colors = len(_COLORS.findall(prompt))
    if n_colors >= 2:
        score += 1
        reasons.append(f"{n_colors} colours")

    n_nouns = _count_noun_phrases(prompt)
    if n_nouns >= _MIN_NOUN_PHRASES:
        score += 1
        reasons.append(f"{n_nouns} nouns")

    return score, reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompts(path: str) -> List[str]:
    p = Path(path)
    if p.suffix == ".jsonl":
        prompts = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    prompts.append(row.get("prompt", row.get("caption", "")))
        return [p for p in prompts if p]
    else:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.input)
    log.info(f"Loaded {len(prompts):,} prompts from {args.input}")

    n_complex = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for prompt in prompts:
            score, reasons = complexity_score(prompt)
            if score >= args.min_score:
                row = {"prompt": prompt, "complexity_score": score, "reasons": reasons}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_complex += 1

    n_simple = len(prompts) - n_complex
    log.info(f"Complex: {n_complex:,}  Simple (skipped): {n_simple:,}  "
             f"({100*n_complex/max(len(prompts),1):.1f}% kept)")
    log.info(f"Output → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,
                        help="prompts.txt or .jsonl with 'prompt' field")
    parser.add_argument("--output",    default="data/complex_prompts.jsonl")
    parser.add_argument("--min-score", type=int, default=2,
                        help="Minimum complexity score to keep (default: 2)")
    args = parser.parse_args()
    main(args)

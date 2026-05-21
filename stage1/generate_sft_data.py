"""
DiffuSVG — Step 1: Generate SFT Training Data
===============================================
Uses the base Qwen2.5-VL-7B-Instruct model to generate its own SVG training
examples (self-supervised). Temperature sampling ensures diversity across runs.

Filters:
  - Must contain <svg>...</svg> tags
  - Must have enough filled shape elements, color variety, and SVG detail
  - Must not duplicate an already saved SVG

Resume-safe: if data/d_sft.jsonl already has rows, appends from where it left off.

Output: data/d_sft.jsonl  (LlamaFactory sharegpt format)
        data/dataset_info.json

Run:
    python generate_sft_data.py --n-samples 500
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sft_data")

DATA_DIR   = Path("data")
OUT_FILE   = DATA_DIR / "d_sft.jsonl"
INFO_FILE  = DATA_DIR / "dataset_info.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT_FILE = Path("../prompts.txt")

MIN_FILLED_ELEMENTS = 8
MIN_UNIQUE_COLORS = 4
MIN_SVG_CHARS = 700

PROMPT_TEMPLATES = [
    # Nature
    "a red apple with a green leaf and a brown stem",
    "a yellow sunflower with a dark center and green stem",
    "a blue butterfly with detailed wing patterns",
    "a green cactus with pink flowers",
    "a purple lavender bouquet tied with a ribbon",
    "a colorful parrot perched on a branch",
    "a red and orange autumn maple leaf",
    "a pink cherry blossom branch with white flowers",
    "a golden wheat field under a blue sky",
    "a snow-capped mountain peak at sunset",
    # Objects
    "a red coffee cup with rising steam",
    "a blue bicycle with yellow wheels",
    "a vintage camera in brown and gold",
    "a green watering can with water drops",
    "a yellow taxi cab on a city street",
    "a colorful kite flying in a blue sky",
    "a wooden treasure chest with golden lock",
    "a red fire hydrant with silver details",
    "a blue umbrella with rain drops",
    "a green mailbox with a red flag",
    # Animals
    "an orange cat sitting with a curled tail",
    "a brown bear eating honey from a jar",
    "a white rabbit with pink ears in grass",
    "a colorful tropical fish in blue water",
    "a red ladybug with black spots on a leaf",
    "a yellow duck swimming on blue water",
    "a green frog sitting on a lily pad",
    "an orange fox running through autumn leaves",
    "a black and white zebra on a green plain",
    "a pink flamingo standing on one leg",
    # Geometric / Abstract
    "a rainbow arc over green hills",
    "a colorful mandala with intricate patterns",
    "a geometric star with alternating red and gold colors",
    "a spiral galaxy with colorful stars",
    "a kaleidoscope pattern in blue and gold",
    "a mosaic of colorful hexagonal tiles",
    "a pinwheel toy with four colorful blades",
    "a set of colorful concentric circles",
    "a decorative Celtic knot in green and gold",
    "a stained glass window pattern in red blue and yellow",
    # Scenes
    "a red barn with a weathervane on a green farm",
    "a sailboat on calm blue water at sunset",
    "a cozy wooden cabin in a snowy forest",
    "a lighthouse on rocky cliffs by the ocean",
    "a tropical beach with palm trees and sun",
    "a city skyline at night with lit windows",
    "a hot air balloon over colorful fields",
    "a park bench under a cherry blossom tree",
    "a windmill in a Dutch landscape with tulips",
    "a campfire under a starry night sky",
    # Food
    "a slice of pizza with red sauce and toppings",
    "a colorful bowl of fruit salad",
    "a layered chocolate cake with strawberries",
    "a cup of ice cream with colorful sprinkles",
    "a glass of lemonade with a yellow straw",
    "a bunch of colorful lollipops on sticks",
    "a steaming bowl of ramen noodles",
    "a stack of fluffy pancakes with maple syrup",
    "a colorful macaron tower in pink and green",
    "a watermelon slice with black seeds",
    # Fantasy / Symbols
    "a golden crown decorated with colorful gems",
    "a red heart with golden wings",
    "a crescent moon with three golden stars",
    "a purple dragon breathing orange fire",
    "a unicorn with a rainbow mane in a field",
    "a castle with colorful flags on its towers",
    "a magic wand with sparkles and stars",
    "a mermaid with a blue tail in the ocean",
    "a wizard hat with golden stars and moon",
    "a phoenix rising from orange flames",
    # Vehicles
    "a vintage red steam locomotive with white smoke",
    "a rocket launching into a starry sky",
    "a vintage wooden sailing ship on blue water",
    "a yellow school bus on a road",
    "a colorful vintage Volkswagen Beetle",
    "a green tractor on a farm field",
    "a blue submarine underwater with fish",
    "a red double-decker bus on a city street",
    "a silver spaceship in outer space",
    "a colorful hot air balloon with a basket",
    # Flowers
    "a red rose with thorny green stem",
    "a blue forget-me-not flower cluster",
    "a yellow and orange marigold",
    "a white daisy with yellow center",
    "a pink peony in full bloom",
    "a purple iris with ruffled petals",
    "a red poppy field under blue sky",
    "an orange tulip in green grass",
    "a white magnolia blossom on a branch",
    "a blue cornflower in a meadow",
    # Complex layered scenes
    "a red barn with green fields below and blue sky above with white clouds",
    "a lighthouse on rocky cliffs with waves crashing below and seagulls above",
    "a city skyline at night with lit windows and a yellow moon above",
    "a sailboat on blue water with an orange sunset sky behind it",
    "a mountain peak with white snow at the top and green pine trees below",
    "a tropical beach with palm trees on the left and a setting sun on the water",
    "a campfire in a forest clearing with stars in the dark sky above",
    "a waterfall over mossy rocks into a blue pool surrounded by green ferns",
    "a desert scene with orange sand dunes and a cactus under a hot yellow sun",
    "a hot air balloon floating above a green and yellow patchwork of fields",
    "a snowy cabin in a pine forest with a chimney and orange light in the windows",
    "a medieval castle on a green hill above a blue moat with colorful flags",
    "a coral reef with orange and yellow fish swimming above blue and purple coral",
    "a rainbow over green rolling hills with a blue river in the valley",
    "a Japanese pagoda on a hill with pink cherry blossom trees on each side",
    "a savanna at sunset with a silhouette of an acacia tree and orange sky",
    "a spring meadow with colorful wildflowers a stream and butterflies",
    "a harbor town with colorful boats on water and houses on a hillside",
    "a winter scene with a frozen pond bare trees and snow on the ground",
    "a rocket launching into a starry sky with a bright exhaust flame below",
]

COMPLEX_FALLBACK_PROMPTS = [
    "a red barn with green fields below and blue sky above with white clouds",
    "a lighthouse on rocky cliffs with waves crashing below and seagulls above",
    "a city skyline at night with lit windows and a yellow moon above",
    "a sailboat on blue water with an orange sunset sky behind it",
    "a mountain peak with white snow at the top and green pine trees below",
    "a tropical beach with palm trees on the left and a setting sun on the water",
    "a campfire in a forest clearing with stars in the dark sky above",
    "a waterfall over mossy rocks into a blue pool surrounded by green ferns",
    "a desert scene with orange sand dunes and a cactus under a hot yellow sun",
    "a hot air balloon floating above a green and yellow patchwork of fields",
    "a snowy cabin in a pine forest with a chimney and orange light in the windows",
    "a medieval castle on a green hill above a blue moat with colorful flags",
    "a coral reef with orange and yellow fish swimming above blue and purple coral",
    "a rainbow over green rolling hills with a blue river in the valley",
    "a Japanese pagoda on a hill with pink cherry blossom trees on each side",
    "a savanna at sunset with a silhouette of an acacia tree and orange sky",
    "a spring meadow with colorful wildflowers a stream and butterflies",
    "a harbor town with colorful boats on water and houses on a hillside",
    "a winter scene with a frozen pond bare trees and snow on the ground",
    "a rocket launching into a starry sky with a bright exhaust flame below",
]


def load_training_prompts(prompt_file: str | Path | None = None) -> list[str]:
    """Load complex scene prompts for SFT/DPO; fall back to built-in scene prompts."""
    path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    prompts: list[str] = []

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if prompt and not prompt.startswith("#"):
                prompts.append(prompt)
        if prompts:
            log.info(f"Loaded {len(prompts)} complex prompts from {path}")
            return prompts

    log.warning(f"Prompt file not found or empty: {path}. Using built-in complex fallback prompts.")
    return COMPLEX_FALLBACK_PROMPTS


def _make_prompt(text: str) -> str:
    return (
        f"You are an expert SVG artist. Create a detailed SVG scene for: '{text}'.\n\n"
        f"STEP 1 — PLAN:\n"
        f"List every object in the scene, its position in the 200×200 canvas, and its color. "
        f"Example: '- Sky: blue rect (0,0)→(200,100)  - Ground: green rect (0,100)→(200,200)'\n\n"
        f"STEP 2 — DRAW:\n"
        f"Generate the SVG based on your plan. Rules:\n"
        f"- viewBox='0 0 200 200'\n"
        f"- Use <rect> for rectangular shapes (sky, ground, buildings, walls)\n"
        f"- Use <circle> or ellipse paths for round shapes (sun, moon, wheels)\n"
        f"- Use <path> for irregular shapes (trees, mountains, waves, animals)\n"
        f"- Every element MUST have a fill= color attribute\n"
        f"- Include AT LEAST 8 distinct elements covering the full canvas\n\n"
        f"Write your PLAN first, then the SVG code starting with <svg."
    )


def _count_filled_elements(svg: str) -> int:
    """Count SVG shape elements that have a fill attribute."""
    return len(re.findall(r'<(?:path|rect|circle|ellipse|polygon|polyline)[^>]+fill', svg))


def _fill_colors(svg: str) -> list[str]:
    """Return normalized visible fill colors from SVG shape elements."""
    colors = re.findall(
        r'<(?:path|rect|circle|ellipse|polygon|polyline|line)\b[^>]*\bfill\s*=\s*["\']([^"\']+)["\']',
        svg,
        flags=re.IGNORECASE,
    )
    return [
        c.strip().lower()
        for c in colors
        if c.strip().lower() not in {"none", "transparent"}
    ]


def _svg_signature(svg: str) -> str:
    """Normalize SVG text enough to catch exact repeated generations."""
    return re.sub(r"\s+", "", svg).lower()


def _quality_failure_reason(svg: str, seen_signatures: set[str]) -> str | None:
    """Return a rejection reason, or None if the SVG passes the quality gate."""
    filled = _count_filled_elements(svg)
    if filled < MIN_FILLED_ELEMENTS:
        return f"fewer than {MIN_FILLED_ELEMENTS} filled elements ({filled})"

    colors = set(_fill_colors(svg))
    if len(colors) < MIN_UNIQUE_COLORS:
        return f"fewer than {MIN_UNIQUE_COLORS} unique fill colors ({len(colors)})"

    if len(svg) < MIN_SVG_CHARS:
        return f"SVG too short ({len(svg)} chars < {MIN_SVG_CHARS})"

    sig = _svg_signature(svg)
    if sig in seen_signatures:
        return "duplicate SVG"

    return None


def _generate_one(prompt: str, model, processor, device) -> str | None:
    """Generate one SVG for a prompt using temperature sampling."""
    import torch
    msg = [{"role": "user", "content": _make_prompt(prompt)}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    try:
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        n = inputs["input_ids"].shape[1]
        raw = processor.tokenizer.decode(ids[0][n:], skip_special_tokens=True).strip()
        m = re.search(r'(<svg[\s>].*?</svg>)', raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None
    except Exception as e:
        log.warning(f"Generation failed: {e}")
        return None


def main(args):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from svg_utils import standardize_svg, is_colorful

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resume: count rows already in the output file
    n_done = 0
    seen_signatures: set[str] = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_done += 1
            try:
                row = json.loads(line)
                svg = next(
                    c["value"]
                    for c in row.get("conversations", [])
                    if c.get("from") == "gpt"
                )
                seen_signatures.add(_svg_signature(svg))
            except Exception:
                pass
        if n_done >= args.n_samples:
            log.info(f"Already have {n_done} samples, nothing to do.")
            return
        log.info(f"Resuming from {n_done} existing samples")

    log.info(f"Loading model: {MODEL_NAME}")
    quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=quant, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Build prompt list from complex scene prompts, shuffle for variety.
    # Stage 1 is intentionally complex-focused so SFT sees background,
    # midground, foreground, and multi-object layouts before DPO/GRPO.
    prompt_source = load_training_prompts(args.prompt_file)
    pool = prompt_source * ((args.n_samples // len(prompt_source)) + 2)
    random.shuffle(pool)
    prompts = pool[:args.n_samples]
    log.info(f"Using {len(prompt_source)} complex source prompts for SFT data generation")

    n_ok = n_done
    n_total_tried = 0

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for prompt in prompts:
            if n_ok >= args.n_samples:
                break

            n_total_tried += 1
            svg_raw = _generate_one(prompt, model, processor, device)
            if svg_raw is None:
                log.info(f"  rejected (no SVG extracted)")
                continue

            svg = standardize_svg(svg_raw) or svg_raw

            if not is_colorful(svg):
                log.info(f"  rejected (not colorful)")
                continue

            failure = _quality_failure_reason(svg, seen_signatures)
            if failure is not None:
                log.info(f"  rejected ({failure})")
                continue

            row = {
                "conversations": [
                    {"from": "human", "value": _make_prompt(prompt)},
                    {"from": "gpt",   "value": svg},
                ]
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            seen_signatures.add(_svg_signature(svg))
            n_ok += 1
            log.info(f"  accepted {n_ok}/{args.n_samples}  (tried {n_total_tried})")
            if n_ok % 10 == 0:
                log.info(f"  checkpoint: {n_ok} saved")

    log.info(f"Done: {n_ok} samples → {OUT_FILE}")
    log.info(f"Accept rate: {n_ok}/{n_total_tried} = {100*n_ok/max(n_total_tried,1):.1f}%")

    # Write dataset_info.json for LlamaFactory
    info = {
        "d_sft": {
            "file_name": "d_sft.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        },
        "d_pref_g": {
            "file_name": "d_pref_g.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    INFO_FILE.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"dataset_info.json → {INFO_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",  type=int, default=1000,
                        help="Total accepted samples to collect")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Kept for CLI compatibility, ignored (always 1)")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE),
                        help="Complex prompt file to use for SFT generation")
    args = parser.parse_args()
    main(args)

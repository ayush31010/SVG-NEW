"""
IntroSVG — SVG Standardization Utilities
=========================================
Implements the D_final standardization pipeline from Section 3.1 of the paper:
  • viewBox  → "0 0 200 200"  (all coordinates scaled accordingly)
  • Commands → M, L, C, A, Z only (absolute, integer coordinates)
  • Attribute order → fill BEFORE d in every <path>
  • Shapes   → <rect>, <circle>, <ellipse> converted to <path>
  • Filter   → remove monochrome, non-renderable, and >8 000-token SVGs
"""

import io
import math
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

# ── SVG namespace ────────────────────────────────────────────────────────────
SVG_NS  = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("",      SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

# ── Monochrome colour patterns (fill / stroke) ────────────────────────────────
_MONO_RE = re.compile(
    r'(?:fill|stroke)\s*=\s*["\']'
    r'(?:#(?:0{3,6}|f{3,6}|[89a-fA-F]{3}(?:[89a-fA-F]{3})?)|'
    r'(?:black|white|gray|grey|none|transparent))["\']',
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r'(?:fill|stroke)\s*=\s*["\']'
    r'(?:#[0-9a-fA-F]{3,6}|rgb\s*\(|hsl\s*\(|[a-zA-Z]+)["\']',
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATH-COMMAND PARSER / CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE  = re.compile(
    r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
)
_CMD_RE  = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]')

_ARGS_PER_CMD = {
    'M': 2, 'm': 2, 'L': 2, 'l': 2,
    'H': 1, 'h': 1, 'V': 1, 'v': 1,
    'C': 6, 'c': 6, 'S': 4, 's': 4,
    'Q': 4, 'q': 4, 'T': 2, 't': 2,
    'A': 7, 'a': 7, 'Z': 0, 'z': 0,
}


def _parse_d(d: str) -> List[Tuple[str, List[float]]]:
    """Tokenise an SVG path `d` string into (command, [args]) pairs."""
    token_re = re.compile(
        r'([MmLlHhVvCcSsQqTtAaZz])|'
        r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
    )
    result: List[Tuple[str, List[float]]] = []
    cur_cmd: Optional[str] = None
    cur_args: List[float] = []

    for m in token_re.finditer(d):
        cmd_tok, num_tok = m.group(1), m.group(2)
        if cmd_tok:
            if cur_cmd is not None:
                result.append((cur_cmd, cur_args))
            cur_cmd = cmd_tok
            cur_args = []
            if _ARGS_PER_CMD.get(cur_cmd, 0) == 0:
                result.append((cur_cmd, []))
                cur_cmd = None
        elif num_tok is not None and cur_cmd is not None:
            cur_args.append(float(num_tok))
            n = _ARGS_PER_CMD.get(cur_cmd, 0)
            if n > 0 and len(cur_args) == n:
                result.append((cur_cmd, cur_args[:]))
                # implicit repetition: M→L, m→l
                cur_cmd = 'L' if cur_cmd == 'M' else ('l' if cur_cmd == 'm' else cur_cmd)
                cur_args = []

    if cur_cmd and cur_args:
        result.append((cur_cmd, cur_args))
    return result


def _to_absolute(
    cmds: List[Tuple[str, List[float]]],
    scale: float = 1.0,
) -> List[Tuple[str, List[float]]]:
    """
    Convert all commands to absolute M, L, C, A, Z.
    Applies `scale` to every coordinate simultaneously (for viewBox normalisation).
    H/V → L, S→C, Q→C (cubic approx), T→C, z→Z.
    """
    result: List[Tuple[str, List[float]]] = []
    cx = cy = 0.0   # current point
    sx = sy = 0.0   # subpath start
    prev_ctrl: Optional[Tuple[float, float]] = None  # for S / T reflection

    def s(v: float) -> float:
        return v * scale

    for cmd, args in cmds:
        up = cmd.upper()
        rel = cmd.islower() and up != 'Z'

        def ax(v):  return (cx + v) * scale if rel else v * scale
        def ay(v):  return (cy + v) * scale if rel else v * scale

        if up == 'M':
            x, y = ax(args[0]), ay(args[1])
            result.append(('M', [x, y]))
            cx, cy = (cx + args[0]) if rel else args[0], (cy + args[1]) if rel else args[1]
            sx, sy = cx, cy
            prev_ctrl = None

        elif up == 'L':
            x, y = ax(args[0]), ay(args[1])
            result.append(('L', [x, y]))
            cx, cy = (cx + args[0]) if rel else args[0], (cy + args[1]) if rel else args[1]
            prev_ctrl = None

        elif up == 'H':
            nx = (cx + args[0]) if rel else args[0]
            result.append(('L', [s(nx), s(cy)]))
            cx = nx
            prev_ctrl = None

        elif up == 'V':
            ny = (cy + args[0]) if rel else args[0]
            result.append(('L', [s(cx), s(ny)]))
            cy = ny
            prev_ctrl = None

        elif up == 'C':
            x1, y1 = ax(args[0]), ay(args[1])
            x2, y2 = ax(args[2]), ay(args[3])
            x,  y  = ax(args[4]), ay(args[5])
            result.append(('C', [x1, y1, x2, y2, x, y]))
            cx, cy = (cx + args[4]) if rel else args[4], (cy + args[5]) if rel else args[5]
            prev_ctrl = (x2 / scale, y2 / scale)

        elif up == 'S':
            x2, y2 = ax(args[0]), ay(args[1])
            x,  y  = ax(args[2]), ay(args[3])
            if prev_ctrl:
                x1 = s(2 * cx - prev_ctrl[0])
                y1 = s(2 * cy - prev_ctrl[1])
            else:
                x1, y1 = s(cx), s(cy)
            result.append(('C', [x1, y1, x2, y2, x, y]))
            cx, cy = (cx + args[2]) if rel else args[2], (cy + args[3]) if rel else args[3]
            prev_ctrl = (args[0] if not rel else cx - args[2] + args[0],
                         args[1] if not rel else cy - args[3] + args[1])

        elif up == 'Q':
            qx1, qy1 = (cx + args[0]) if rel else args[0], (cy + args[1]) if rel else args[1]
            nx,  ny  = (cx + args[2]) if rel else args[2], (cy + args[3]) if rel else args[3]
            # quadratic → cubic
            bx1 = cx + 2/3 * (qx1 - cx);  by1 = cy + 2/3 * (qy1 - cy)
            bx2 = nx + 2/3 * (qx1 - nx);  by2 = ny + 2/3 * (qy1 - ny)
            result.append(('C', [s(bx1), s(by1), s(bx2), s(by2), s(nx), s(ny)]))
            cx, cy = nx, ny
            prev_ctrl = (qx1, qy1)

        elif up == 'T':
            nx, ny = (cx + args[0]) if rel else args[0], (cy + args[1]) if rel else args[1]
            if prev_ctrl:
                qx1 = 2 * cx - prev_ctrl[0];  qy1 = 2 * cy - prev_ctrl[1]
            else:
                qx1, qy1 = cx, cy
            bx1 = cx + 2/3 * (qx1 - cx);  by1 = cy + 2/3 * (qy1 - cy)
            bx2 = nx + 2/3 * (qx1 - nx);  by2 = ny + 2/3 * (qy1 - ny)
            result.append(('C', [s(bx1), s(by1), s(bx2), s(by2), s(nx), s(ny)]))
            cx, cy = nx, ny
            prev_ctrl = (qx1, qy1)

        elif up == 'A':
            rx, ry, rot, la, sw = args[0], args[1], args[2], args[3], args[4]
            nx = (cx + args[5]) if rel else args[5]
            ny = (cy + args[6]) if rel else args[6]
            result.append(('A', [s(rx), s(ry), rot, int(la), int(sw), s(nx), s(ny)]))
            cx, cy = nx, ny
            prev_ctrl = None

        elif up == 'Z':
            result.append(('Z', []))
            cx, cy = sx, sy
            prev_ctrl = None

    return result


def _round_cmds(cmds: List[Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
    """Round all coordinates to integers."""
    out = []
    for cmd, args in cmds:
        if cmd == 'A':
            out.append(('A', [
                round(args[0]), round(args[1]),
                round(args[2]),
                int(args[3]), int(args[4]),
                round(args[5]), round(args[6]),
            ]))
        elif not args:
            out.append((cmd, []))
        else:
            out.append((cmd, [round(v) for v in args]))
    return out


def _cmds_to_d(cmds: List[Tuple[str, List[float]]]) -> str:
    parts = []
    for cmd, args in cmds:
        if args:
            parts.append(cmd + ' ' + ' '.join(
                str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
                for v in args
            ))
        else:
            parts.append(cmd)
    return ' '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAPE → PATH CONVERSIONS
# ─────────────────────────────────────────────────────────────────────────────

def _float_attr(el: ET.Element, name: str, default: float = 0.0) -> float:
    return float(el.get(name, default))

_K = 0.5522847498   # 4/3·tan(π/8) for circle → cubic bezier

def _rect_to_d(el: ET.Element, scale: float) -> str:
    x  = _float_attr(el, 'x') * scale
    y  = _float_attr(el, 'y') * scale
    w  = _float_attr(el, 'width')  * scale
    h  = _float_attr(el, 'height') * scale
    rx = _float_attr(el, 'rx') * scale
    ry = _float_attr(el, 'ry', rx) * scale
    if rx == 0:
        return (f"M {round(x)} {round(y)} L {round(x+w)} {round(y)} "
                f"L {round(x+w)} {round(y+h)} L {round(x)} {round(y+h)} Z")
    return (f"M {round(x+rx)} {round(y)} L {round(x+w-rx)} {round(y)} "
            f"A {round(rx)} {round(ry)} 0 0 1 {round(x+w)} {round(y+ry)} "
            f"L {round(x+w)} {round(y+h-ry)} "
            f"A {round(rx)} {round(ry)} 0 0 1 {round(x+w-rx)} {round(y+h)} "
            f"L {round(x+rx)} {round(y+h)} "
            f"A {round(rx)} {round(ry)} 0 0 1 {round(x)} {round(y+h-ry)} "
            f"L {round(x)} {round(y+ry)} "
            f"A {round(rx)} {round(ry)} 0 0 1 {round(x+rx)} {round(y)} Z")

def _circle_to_d(el: ET.Element, scale: float) -> str:
    cx = _float_attr(el, 'cx') * scale
    cy = _float_attr(el, 'cy') * scale
    r  = _float_attr(el, 'r')  * scale
    k  = _K * r
    return (f"M {round(cx)} {round(cy-r)} "
            f"C {round(cx+k)} {round(cy-r)} {round(cx+r)} {round(cy-k)} {round(cx+r)} {round(cy)} "
            f"C {round(cx+r)} {round(cy+k)} {round(cx+k)} {round(cy+r)} {round(cx)} {round(cy+r)} "
            f"C {round(cx-k)} {round(cy+r)} {round(cx-r)} {round(cy+k)} {round(cx-r)} {round(cy)} "
            f"C {round(cx-r)} {round(cy-k)} {round(cx-k)} {round(cy-r)} {round(cx)} {round(cy-r)} Z")

def _ellipse_to_d(el: ET.Element, scale: float) -> str:
    cx = _float_attr(el, 'cx') * scale
    cy = _float_attr(el, 'cy') * scale
    rx = _float_attr(el, 'rx') * scale
    ry = _float_attr(el, 'ry') * scale
    kx, ky = _K * rx, _K * ry
    return (f"M {round(cx)} {round(cy-ry)} "
            f"C {round(cx+kx)} {round(cy-ry)} {round(cx+rx)} {round(cy-ky)} {round(cx+rx)} {round(cy)} "
            f"C {round(cx+rx)} {round(cy+ky)} {round(cx+kx)} {round(cy+ry)} {round(cx)} {round(cy+ry)} "
            f"C {round(cx-kx)} {round(cy+ry)} {round(cx-rx)} {round(cy+ky)} {round(cx-rx)} {round(cy)} "
            f"C {round(cx-rx)} {round(cy-ky)} {round(cx-kx)} {round(cy-ry)} {round(cx)} {round(cy-ry)} Z")

def _line_to_d(el: ET.Element, scale: float) -> str:
    x1 = round(_float_attr(el, 'x1') * scale)
    y1 = round(_float_attr(el, 'y1') * scale)
    x2 = round(_float_attr(el, 'x2') * scale)
    y2 = round(_float_attr(el, 'y2') * scale)
    return f"M {x1} {y1} L {x2} {y2}"

def _polyline_to_d(el: ET.Element, scale: float, close: bool = False) -> str:
    pts_raw = re.findall(r'[-+]?\d*\.?\d+', el.get('points', ''))
    pts = [round(float(v) * scale) for v in pts_raw]
    if len(pts) < 4:
        return ""
    pairs = [f"{pts[i]} {pts[i+1]}" for i in range(0, len(pts) - 1, 2)]
    d = "M " + pairs[0] + " L " + ' L '.join(pairs[1:])
    return d + (" Z" if close else "")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN STANDARDISATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def standardize_svg(svg_code: str) -> Optional[str]:
    """
    Apply the full D_final standardisation pipeline to one SVG string.
    Returns the standardised SVG string, or None if it cannot be standardised.
    """
    svg_code = svg_code.strip()
    if not svg_code:
        return None

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        # Strip XML declaration to avoid ET complaints
        svg_code = re.sub(r'<\?xml[^>]*\?>', '', svg_code).strip()
        # Inject namespace if missing so ET parses correctly
        if 'xmlns=' not in svg_code:
            svg_code = svg_code.replace('<svg', f'<svg xmlns="{SVG_NS}"', 1)
        root = ET.fromstring(svg_code)
    except ET.ParseError:
        return None

    # ── Compute viewBox scale factor ─────────────────────────────────────────
    vb = root.get('viewBox', '').strip()
    scale = 1.0
    if vb:
        parts = re.findall(r'[-+]?\d*\.?\d+', vb)
        if len(parts) >= 4:
            vb_w = float(parts[2])
            vb_h = float(parts[3])
            if vb_w > 0 and vb_h > 0:
                scale = min(200.0 / vb_w, 200.0 / vb_h)
    elif root.get('width') and root.get('height'):
        try:
            w = float(re.sub(r'[^\d.]', '', root.get('width', '200')))
            h = float(re.sub(r'[^\d.]', '', root.get('height', '200')))
            scale = min(200.0 / w, 200.0 / h) if w > 0 and h > 0 else 1.0
        except ValueError:
            pass

    root.set('viewBox', '0 0 200 200')
    # Remove width/height attributes — viewBox alone is canonical
    root.attrib.pop('width',  None)
    root.attrib.pop('height', None)

    # ── Walk all elements, convert shapes, standardise paths ─────────────────
    ns = {'svg': SVG_NS}
    tag_map = {
        f'{{{SVG_NS}}}rect':     ('rect',     _rect_to_d),
        f'{{{SVG_NS}}}circle':   ('circle',   _circle_to_d),
        f'{{{SVG_NS}}}ellipse':  ('ellipse',  _ellipse_to_d),
        f'{{{SVG_NS}}}line':     ('line',     _line_to_d),
        f'{{{SVG_NS}}}polyline': ('polyline', lambda e, s: _polyline_to_d(e, s, False)),
        f'{{{SVG_NS}}}polygon':  ('polygon',  lambda e, s: _polyline_to_d(e, s, True)),
    }
    tag_path = f'{{{SVG_NS}}}path'

    new_paths: List[Tuple[ET.Element, ET.Element, str]] = []  # (parent, old, new)

    def _walk(parent: ET.Element):
        for child in list(parent):
            if child.tag in tag_map:
                _, fn = tag_map[child.tag]
                d = fn(child, scale)
                if not d:
                    parent.remove(child)
                    continue
                path_el = ET.Element(f'{{{SVG_NS}}}path')
                # Preserve visual attributes (fill, stroke, opacity, etc.)
                KEEP = {'fill', 'stroke', 'stroke-width', 'opacity',
                        'fill-opacity', 'stroke-opacity', 'fill-rule',
                        'stroke-linecap', 'stroke-linejoin', 'transform'}
                attribs = {k: v for k, v in child.attrib.items()
                           if k in KEEP or k.startswith('class') or k.startswith('style')}
                # fill before d — insert fill first, then d
                fill = attribs.pop('fill', 'none')
                path_el.set('fill', fill)
                for k, v in attribs.items():
                    path_el.set(k, v)
                path_el.set('d', d)
                new_paths.append((parent, child, path_el))
            elif child.tag == tag_path:
                _standardise_path(child, scale)
            _walk(child)

    def _standardise_path(el: ET.Element, sc: float):
        d_raw = el.get('d', '').strip()
        if not d_raw:
            return
        try:
            cmds  = _parse_d(d_raw)
            cmds  = _to_absolute(cmds, sc)
            cmds  = _round_cmds(cmds)
            d_new = _cmds_to_d(cmds)
        except Exception:
            return
        # Rebuild attribs: fill first, then rest, then d last
        fill = el.get('fill', 'none')
        old_attribs = dict(el.attrib)
        el.attrib.clear()
        el.set('fill', fill)
        for k, v in old_attribs.items():
            if k not in ('fill', 'd'):
                el.set(k, v)
        el.set('d', d_new)

    _walk(root)

    # Apply shape replacements
    for parent, old, new in new_paths:
        try:
            idx = list(parent).index(old)
            parent.remove(old)
            parent.insert(idx, new)
        except (ValueError, Exception):
            pass

    # ── Serialise ────────────────────────────────────────────────────────────
    try:
        out = ET.tostring(root, encoding='unicode')
        # Clean up namespace prefixes ET adds
        out = re.sub(r' ns\d+:href', ' href', out)
        return out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FILTER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_colorful(svg_code: str) -> bool:
    """Return True if SVG contains at least one non-monochrome colour."""
    colors = _COLOR_RE.findall(svg_code)
    mono   = _MONO_RE.findall(svg_code)
    return (len(colors) - len(mono)) > 0


def render_to_png(svg_code: str, size: int = 200) -> Optional[bytes]:
    """Render SVG → PNG bytes via cairosvg. Returns None on failure."""
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_code.encode(),
            output_width=size,
            output_height=size,
        )
    except Exception:
        return None


def is_renderable(svg_code: str) -> bool:
    return render_to_png(svg_code, size=64) is not None


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PROMPT BUILDERS  (match training data format)
# ─────────────────────────────────────────────────────────────────────────────

def gen_prompt(prompt: str) -> str:
    return (
        f"You are an expert SVG artist. Create a detailed SVG scene for: '{prompt}'.\n\n"
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


def critic_prompt(prompt: str) -> str:
    return (
        f'You are a professional SVG design critic. Please analyze the input '
        f'AI-generated SVG draft according to the "Original Design Prompt".\n\n'
        f'**Original Design Prompt**: "{prompt}"\n\n'
        f'Your task is to output a structured critique report, strictly following '
        f'the JSON format: {{"score": <0-10>, "critique": "<issues>", "suggestions": "<fixes>"}}'
    )


def correction_prompt(prompt: str, svg: str, critique: dict) -> str:
    import json
    return (
        f"Please analyze all the information provided below and generate a final, "
        f"high-quality SVG code.\n\n"
        f"The original design prompt was: {prompt}\n\n"
        f"A draft SVG code is:\n{svg}\n\n"
        f"An expert critique and suggestions of this draft is:\n"
        f"{json.dumps(critique, ensure_ascii=False)}"
    )

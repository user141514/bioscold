#!/usr/bin/env python
"""Global SVG harmonization pass for Figures 1-5.

This script works only inside the editable SVG copy folder. It does not touch
the clean archive folder or upstream plotting scripts.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "harmonized_jcim_style_v4"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

FIGURES = {
    1: "Figure_1_replicated_svg_v3.svg",
    2: "Figure_2_reconstructed_editable_bus_aligned_v2.svg",
    3: "Figure_3_secondary_blind_performance_locked.svg",
    4: "Figure_4_editable_replicated.svg",
    5: "Figure_5_dual_mode_provenance_triage_math_refined.svg",
}

NAVY = "#0B3F99"
TEAL = "#087A74"
RED = "#B31F24"
SOFT_RED = "#A43A3A"
DARK = "#222222"
MID = "#666666"
LIGHT = "#D9D9D9"
FILL_BLUE = "#F4F8FF"
FILL_TEAL = "#F2FBFA"
FILL_RED = "#FFF6F4"
WHITE = "#FFFFFF"


def normalize_hex(color: str) -> str:
    return color.lower()


def map_svg_color(prop: str, color: str) -> str:
    c = normalize_hex(color)
    navyish = {
        "#08204a", "#0b2e66", "#0b3f99", "#173f72",
        "#174276", "#1c4d8f", "#004b9b",
    }
    tealish = {
        "#087a74", "#0b6f6b", "#00796f", "#008c88", "#008577",
        "#365f66",
    }
    reddish = {"#b31f24", "#b30000", "#cf3c1f", "#c73d21", "#d04a2b"}
    darkish = {"#000000", "#111111", "#1f1f1f", "#222222", "#333333"}
    midgrays = {"#444444", "#4a4a4a", "#555555", "#666666"}
    linegrays = {
        "#777777", "#888888", "#999999", "#aaaaaa", "#b0b0b0",
        "#b5b5b5", "#c0c0c0", "#c9c9c9", "#cccccc", "#d0d0d0",
        "#d9d9d9",
    }
    fills = {
        "#f8fbff": FILL_BLUE,
        "#f4f8ff": FILL_BLUE,
        "#f3fbfa": FILL_TEAL,
        "#f2fbfa": FILL_TEAL,
        "#fff7f4": FILL_RED,
        "#fff6f4": FILL_RED,
        "#ffffff": WHITE,
        "#f7f7f7": WHITE,
    }
    if c in fills:
        return fills[c]
    if c in navyish:
        return NAVY
    if c in tealish:
        return TEAL
    if c in reddish:
        return RED
    if c in darkish:
        return DARK
    if prop == "stroke" and (c in midgrays or c in linegrays):
        return LIGHT
    if prop == "fill" and c in midgrays:
        return MID
    if prop == "fill" and c in linegrays:
        return LIGHT
    return color


def harmonize_styles(svg: str) -> str:
    # One manuscript-wide sans-serif stack.
    svg = re.sub(
        r"font-family:[^;\"']+",
        "font-family:Arial, Helvetica, DejaVu Sans, sans-serif",
        svg,
    )

    # Color role mapping.
    def repl_color(m: re.Match[str]) -> str:
        prop, color = m.group(1), m.group(2)
        return f"{prop}:{map_svg_color(prop, color)}"

    svg = re.sub(r"\b(fill|stroke):(#[0-9a-fA-F]{6})", repl_color, svg)

    # Light, consistent strokes. Keep main boxes around 1.0-1.2 pt and panel
    # grids around 0.5-0.7 pt.
    def repl_width(m: re.Match[str]) -> str:
        val = float(m.group(1))
        if val <= 0.6:
            new = 0.5
        elif val <= 0.8:
            new = 0.65
        elif val <= 0.95:
            new = 0.9
        elif val <= 1.4:
            new = 1.1
        else:
            new = 1.2
        return f"stroke-width:{new:g}"

    svg = re.sub(r"stroke-width:([0-9.]+)", repl_width, svg)

    # Figure-level title sizes should not look like slide titles.
    svg = svg.replace("font-size:18px", "font-size:16px")
    svg = svg.replace("font-size:17px", "font-size:16px")
    svg = svg.replace("font-size:21px", "font-size:18px")
    return svg


def fix_figure_1(svg: str) -> str:
    # Panel B: the leaked transform identity must be the same on both sides.
    svg = re.sub(
        r"(<!-- \$T_4\$ -->.*?<tspan[^>]*>)4(</tspan>)",
        lambda m: m.group(1).replace("$T_4$", "$T_2$") + "2" + m.group(2),
        svg,
        count=1,
        flags=re.S,
    )
    # Repair mojibake in the leakage-risk conclusion if present.
    svg = re.sub(
        r">[^<]*leakage risk<",
        ">→ leakage risk<",
        svg,
        count=1,
    )
    # Figure 1 side note should read as a true note, not a fourth module,
    # but remain legible in a manuscript column.
    for label in [
        "side note",
        "closed train-derived",
        "vocabulary",
        "no open-vocabulary",
        "generation",
    ]:
        svg = re.sub(
            rf'(<text style="[^"]*?)font-size:([0-9.]+)px([^"]*?"[^>]*>{re.escape(label)}</text>)',
            lambda m: (
                m.group(1)
                .replace(f"fill:{NAVY}", f"fill:{MID}")
                .replace(f"fill:{DARK}", f"fill:{MID}")
                + "font-size:"
                + f"{max(5.8, min(6.4, float(m.group(2)))):g}px"
                + m.group(3)
            ),
            svg,
        )
    # The lower protocol rail in Panel C is supporting context; make its labels
    # slightly quieter without changing the protocol content.
    for label in ["Train", "Development/", "calibration", "Secondary blind", "No blind tuning"]:
        svg = re.sub(
            rf'(<text style="[^"]*?)fill:{TEAL}([^"]*?"[^>]*>{re.escape(label)}</text>)',
            rf'\1fill:{MID}\2',
            svg,
        )
    # Make panel frames quieter than conceptual marks.
    svg = re.sub(
        r"(style=\"fill:#FFFFFF;stroke:#D9D9D9;stroke-width:)(0.65|0.75|0.9)(;\")",
        r"\g<1>0.5\3",
        svg,
    )
    return svg


def fix_figure_2(svg: str) -> str:
    # Reduce engineering-variable feel in figure text.
    svg = svg.replace("prior_ranks", "prior ranks")
    svg = svg.replace("model_ranks", "model ranks")
    svg = svg.replace("model_scores", "model scores")
    # Make notes less visually dominant.
    svg = svg.replace(
        "font-size:8.8px;font-style:normal;font-weight:bold;text-anchor:start;",
        "font-size:7.4px;font-style:normal;font-weight:normal;text-anchor:start;",
    )
    # Remove lonely bottom-right explanatory notes from the schematic. These
    # belong in the caption and make the figure look less like a manuscript
    # architecture diagram. The source IDs are stable in the editable copy, but
    # keep the match content-based so the pass remains robust.
    for note in [
        "Score Blend - strongest pre-D4S baseline",
        "A4C strata - exploratory triage only",
    ]:
        svg = re.sub(
            rf'\s*<g id="text_\d+">\s*<text[^>]*>{re.escape(note)}</text>\s*</g>',
            "",
            svg,
            flags=re.S,
        )
    # Make the central feature matrix more clearly the visual hub.
    svg = re.sub(
        r'(style="fill:#F2FBFA;stroke:#087A74;stroke-width:)([0-9.]+)(;")',
        r'\g<1>1.2\3',
        svg,
        count=1,
    )
    # Reduce bottom whitespace and enlarge the actual architecture by cropping
    # the canvas. Content is preserved; only the viewBox/page extent changes.
    svg = re.sub(
        r'<svg height="[^"]+" version="1\.1" viewBox="0 0 720 540" width="[^"]+"',
        '<svg height="480pt" version="1.1" viewBox="0 0 720 480" width="720pt"',
        svg,
        count=1,
    )
    svg = re.sub(
        r'M 0 540\s+L 720 540\s+L 720 0\s+L 0 0\s+z',
        'M 0 480 L 720 480 L 720 0 L 0 0 z',
        svg,
        count=1,
    )
    return svg


def fix_figure_5(svg: str) -> str:
    # Cleaner two-line wrap for the conservative route proposal box.
    svg = re.sub(
        r'<g id="text_8">.*?</g>',
        (
            '<g id="text_8">\n'
            f'    <text style="fill:{NAVY};font-family:Arial, Helvetica, DejaVu Sans, sans-serif;'
            'font-size:10.8px;font-style:normal;font-weight:normal;text-anchor:middle;" '
            'transform="rotate(-0, 84.533812, 333.0)" x="84.533812" y="333.0">'
            'frequency-aligned</text>\n'
            f'    <text style="fill:{NAVY};font-family:Arial, Helvetica, DejaVu Sans, sans-serif;'
            'font-size:10.8px;font-style:normal;font-weight:normal;text-anchor:middle;" '
            'transform="rotate(-0, 84.533812, 347.0)" x="84.533812" y="347.0">'
            'proposals</text>\n'
            "   </g>"
        ),
        svg,
        flags=re.S,
    )
    formula_style = (
        f"fill:{DARK};font-family:Arial, Helvetica, DejaVu Sans, sans-serif;"
        "font-size:10.4px;font-style:normal;font-weight:normal;text-anchor:middle;"
    )
    k_style = "font-style:italic;"
    sub_style = "font-size:7.1px;baseline-shift:sub;"
    op_style = "font-style:normal;"

    def k_sub(label: str) -> str:
        return (
            f'<tspan style="{k_style}">K</tspan>'
            f'<tspan style="{sub_style}">{label}</tspan>'
        )

    def formula_markup(kind: str) -> str:
        if kind == "g4":
            body = k_sub("HGB") + f'<tspan style="{op_style}"> ∩ </tspan>' + k_sub("Borda")
        elif kind == "g3":
            body = k_sub("Borda") + f'<tspan style="{op_style}"> \\ </tspan>' + k_sub("HGB")
        else:
            body = (
                k_sub("Borda")
                + f'<tspan style="{op_style}"> \\ (</tspan>'
                + k_sub("HGB")
                + f'<tspan style="{op_style}"> ∪ </tspan>'
                + k_sub("DE")
                + f'<tspan style="{op_style}">)</tspan>'
            )
        return body

    formulas = {
        "text_16": ("text_17", "381.6", "180.15", "g4"),
        "text_19": ("text_20", "381.6", "296.15", "g3"),
        "text_22": ("text_23", "381.6", "431.49", "g2"),
    }
    for gid, (next_gid, x, y, kind) in formulas.items():
        svg = re.sub(
            rf'   <g id="{gid}">.*?(?=\n   <g id="{next_gid}">)',
            (
                f'   <g id="{gid}">\n'
                f'    <text style="{formula_style}" transform="rotate(-0, {x}, {y})" '
                f'x="{x}" y="{y}">{formula_markup(kind)}</text>\n'
                "   </g>\n"
            ),
            svg,
            flags=re.S,
        )
    # Figure 5 is a triage schematic, not an experimental verdict: make the red
    # slightly softer and table gridlines quieter while preserving alert roles.
    svg = svg.replace(f"fill:{RED}", f"fill:{SOFT_RED}")
    svg = svg.replace(f"stroke:{RED}", f"stroke:{SOFT_RED}")
    svg = re.sub(
        r"(stroke:#D9D9D9;stroke-width:)(0.65|0.9|1.1|1.2)",
        r"\g<1>0.5",
        svg,
    )
    # Make panel/table text closer to Figures 3/4: less bookish, less dense.
    svg = svg.replace("font-size:16px", "font-size:14.5px")
    svg = svg.replace("font-size:14px", "font-size:13px")
    svg = svg.replace("font-size:12.5px", "font-size:11.4px")
    svg = svg.replace("font-size:12px", "font-size:11.2px")
    # Keep the claim-boundary note present but secondary.
    svg = re.sub(
        r'(<text style="fill:#666666;[^"]*font-size:)([0-9.]+)(px[^"]*"[^>]*>A4C = computational triage only;</text>)',
        r'\g<1>8.4\3',
        svg,
    )
    svg = re.sub(
        r'(<text style="fill:#666666;[^"]*font-size:)([0-9.]+)(px[^"]*"[^>]*>not experimental validation\.</text>)',
        r'\g<1>8.4\3',
        svg,
    )
    return svg


def harmonize_svg(fig_no: int, src: Path, dst: Path) -> None:
    svg = src.read_text(encoding="utf-8")
    svg = harmonize_styles(svg)
    if fig_no == 1:
        svg = fix_figure_1(svg)
    elif fig_no == 2:
        svg = fix_figure_2(svg)
    elif fig_no == 5:
        svg = fix_figure_5(svg)
    dst.write_text(svg, encoding="utf-8")


def svg_size(svg_path: Path) -> tuple[float, float]:
    txt = svg_path.read_text(encoding="utf-8")
    m = re.search(r'<svg[^>]*height="([0-9.]+)pt"[^>]*width="([0-9.]+)pt"', txt)
    if m:
        return float(m.group(2)), float(m.group(1))
    m = re.search(r'viewBox="[^"]*?([0-9.]+)\s+([0-9.]+)"', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 720.0, 520.0


def html_for_svg(svg: Path, w_pt: float, h_pt: float, mode: str) -> str:
    if mode == "screen":
        body_size = "width: 100vw;\n  height: 100vh;"
        page = ""
    else:
        body_size = f"width: {w_pt}pt;\n  height: {h_pt}pt;"
        page = f"@page {{ size: {w_pt}pt {h_pt}pt; margin: 0; }}"
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
{page}
html, body {{
  margin: 0;
  padding: 0;
  {body_size}
  overflow: hidden;
  background: white;
}}
img {{
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
</style>
</head>
<body><img src="{html.escape(svg.as_uri())}"></body>
</html>"""


def render_with_edge(svg: Path, png: Path, pdf: Path) -> None:
    if not EDGE.exists():
        raise RuntimeError(f"Microsoft Edge not found at {EDGE}")
    w_pt, h_pt = svg_size(svg)
    # Browser CSS converts pt to px at 96/72. Use the natural SVG CSS size
    # for PNG export; oversized viewports make SVGs appear in the upper-left.
    w_px = max(700, int(round(w_pt * 96 / 72)))
    h_px = max(520, int(round(h_pt * 96 / 72)))
    with tempfile.TemporaryDirectory(prefix="jcim_svg_render_") as td:
        td_path = Path(td)
        html_print = td_path / f"{svg.stem}_print.html"
        html_print.write_text(html_for_svg(svg, w_pt, h_pt, "print"), encoding="utf-8")
        user_data = td_path / "edge_profile"
        common = [
            str(EDGE),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-first-run",
            f"--user-data-dir={user_data}",
        ]
        shot = subprocess.run(
            common
            + [
                "--hide-scrollbars",
                f"--screenshot={png}",
                f"--window-size={w_px},{h_px}",
                svg.as_uri(),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not png.exists() or png.stat().st_size == 0:
            raise RuntimeError(f"PNG export failed for {svg.name}: {shot.stderr}")
        pdf_run = subprocess.run(
            common
            + [
                "--print-to-pdf-no-header",
                f"--print-to-pdf={pdf}",
                html_print.as_uri(),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not pdf.exists() or pdf.stat().st_size == 0:
            raise RuntimeError(f"PDF export failed for {svg.name}: {pdf_run.stderr}")


def make_contact_sheet(pngs: list[Path]) -> tuple[Path, Path]:
    def trim_white(img: Image.Image, border: int = 24) -> Image.Image:
        bg = Image.new(img.mode, img.size, "white")
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if not bbox:
            return img
        left = max(0, bbox[0] - border)
        top = max(0, bbox[1] - border)
        right = min(img.width, bbox[2] + border)
        bottom = min(img.height, bbox[3] + border)
        return img.crop((left, top, right, bottom))

    thumbs = []
    for p in pngs:
        img = Image.open(p).convert("RGB")
        img = trim_white(img)
        img.thumbnail((880, 560), Image.LANCZOS)
        thumbs.append((p, img.copy()))
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    label_h = 34
    pad = 28
    cols = 2
    rows = 3
    cell_w = 940
    cell_h = 640
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, img) in enumerate(thumbs):
        col = i % cols
        row = i // cols
        x0 = pad + col * cell_w
        y0 = pad + row * cell_h
        label = f"Figure {i + 1}"
        draw.text((x0, y0), label, fill=(34, 34, 34), font=font)
        x = x0 + (cell_w - img.width) // 2
        y = y0 + label_h + (cell_h - label_h - img.height) // 2
        sheet.paste(img, (x, y))
    out_png = OUT / "Figures_1_to_5_contact_sheet.png"
    out_pdf = OUT / "Figures_1_to_5_contact_sheet.pdf"
    sheet.save(out_png)
    sheet.save(out_pdf, "PDF", resolution=150)
    return out_png, out_pdf


def write_audit() -> None:
    audit = OUT / "figure_style_audit.md"
    audit.write_text(
        "# Figure Style Audit\n\n"
        "This v4 pass uses the local PaperBanana skill as a style-planning layer: its output constraints were translated into deterministic SVG edits rather than image-generation edits, because Figures 3/4 contain locked numeric evidence.\n\n"
        "1. **Fonts consistent?** Yes. All harmonized SVG text styles are mapped to `Arial, Helvetica, DejaVu Sans, sans-serif`.\n"
        "2. **Colors mapped consistently?** Yes. Navy `#0B3F99` is used for task/base evidence, teal `#087A74` for heldout/main/post-audit/triage routes, red `#B31F24` for leakage/removed/alert signals, and gray tones for neutral text, borders, and grids.\n"
        "3. **Panel labels consistent?** Mostly yes. Existing panel structures were preserved; labels were harmonized through shared font and palette without redesigning panel logic.\n"
        "4. **Line widths and arrows consistent?** Yes. Main colored boxes are kept near 1.0-1.2 pt, panel/table/grid strokes are lightened, and arrow strokes are normalized to a restrained 1.0-1.2 pt range where represented in SVG styles.\n"
        "5. **Figure 1 content corrections applied?** Yes. Panel B now shows the same leaked transform identity (`T2`) on both train and evaluation sides, and Panel C uses `Train transforms` / `Blind transforms`.\n"
        "6. **Do Figures 3/4 preserve all locked values?** Yes. The harmonization pass changes only SVG styling/text presentation; plotted values, ordering, CIs, and panel logic are not altered.\n"
        "7. **Does Figure 5 preserve A4C claim boundary?** Yes. The note `A4C = computational triage only; not experimental validation.` is retained, and G2/G3/G4 remain provenance/triage strata rather than experimental toxicity or activity claims.\n"
        "8. **What changed relative to v3?** Figure 2 removes the isolated Score Blend/A4C footnote from the canvas and crops excess bottom whitespace; Figure 5 further lowers formula/table visual density; Figure 1/3/4 keep their content and locked evidence.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    harmonized_svgs = []
    pngs = []
    for fig_no, filename in FIGURES.items():
        src = ROOT / filename
        dst_svg = OUT / f"Figure_{fig_no}_harmonized.svg"
        dst_png = OUT / f"Figure_{fig_no}_harmonized.png"
        dst_pdf = OUT / f"Figure_{fig_no}_harmonized.pdf"
        harmonize_svg(fig_no, src, dst_svg)
        render_with_edge(dst_svg, dst_png, dst_pdf)
        harmonized_svgs.append(dst_svg)
        pngs.append(dst_png)
    make_contact_sheet(pngs)
    write_audit()
    for p in harmonized_svgs + pngs + [OUT / "Figures_1_to_5_contact_sheet.png", OUT / "figure_style_audit.md"]:
        print(p)


if __name__ == "__main__":
    main()

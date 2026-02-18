#!/usr/bin/env python3
"""Generate and optimize 2D particle lock-and-key geometries.

This script supports two workflows:
1) Export baseline designs (`--mode baseline`)
2) Auto-search edge templates/parameters and export best candidates (`--mode search`)

All dimensions are in micrometers.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EdgeFeatures:
    top: bool = False
    right: bool = False
    bottom: bool = False
    left: bool = False


@dataclass(frozen=True)
class ParticleSpec:
    width: float = 20.0
    height: float = 20.0
    tab_depth: float = 1.6
    tab_width: float = 4.0
    corner_chamfer: float = 1.1


@dataclass(frozen=True)
class Candidate:
    template: str
    tab_depth: float
    tab_width: float
    ramp: float
    asymmetry: float
    lock_strength: float
    approach_window: float
    score: float


def _linspace(start: float, end: float, n: int) -> list[float]:
    if n <= 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def edge_profile(
    length: float,
    outward: bool,
    feature: bool,
    template: str,
    tab_width: float,
    tab_depth: float,
    ramp: float,
    asymmetry: float,
) -> list[tuple[float, float]]:
    """Build a top-edge profile from left->right.

    y<0 is outward, y>0 is inward for the top edge's local frame.
    """
    if not feature:
        return [(0.0, 0.0), (length, 0.0)]

    sign = -1.0 if outward else 1.0
    d = sign * tab_depth

    if template == "single_tooth":
        w1 = 0.5 * (length - tab_width)
        w2 = 0.5 * (length + tab_width)
        return [
            (0.0, 0.0),
            (w1, 0.0),
            (w1 + ramp, d),
            (w2 - ramp, d),
            (w2, 0.0),
            (length, 0.0),
        ]

    if template == "double_tooth_asym":
        gap = 1.2 + 0.6 * asymmetry
        t1 = tab_width * (0.95 - 0.1 * asymmetry)
        t2 = tab_width * (0.75 + 0.2 * asymmetry)
        left = 0.25 * length - 0.5 * t1
        mid = 0.5 * length
        right = mid + gap + 0.5 * t2
        left2 = mid - gap - 0.5 * t1
        return [
            (0.0, 0.0),
            (left, 0.0),
            (left + ramp, d),
            (left + t1 - ramp, d),
            (left + t1, 0.0),
            (left2, 0.0),
            (left2 + ramp, d),
            (left2 + t1 - ramp, d),
            (left2 + t1, 0.0),
            (mid + gap - 0.5 * t2, 0.0),
            (mid + gap - 0.5 * t2 + ramp, d),
            (right - ramp, d),
            (right, 0.0),
            (length, 0.0),
        ]

    if template == "dovetail":
        w1 = 0.5 * (length - tab_width)
        w2 = 0.5 * (length + tab_width)
        neck = 0.45 * tab_width
        return [
            (0.0, 0.0),
            (w1, 0.0),
            (w1 + ramp, 0.55 * d),
            (0.5 * (w1 + w2) - 0.5 * neck, d),
            (0.5 * (w1 + w2) + 0.5 * neck, d),
            (w2 - ramp, 0.55 * d),
            (w2, 0.0),
            (length, 0.0),
        ]

    raise ValueError(f"Unknown template: {template}")


def profile_depth_function(
    length: float,
    template: str,
    tab_width: float,
    tab_depth: float,
    ramp: float,
    asymmetry: float,
    samples: int = 301,
) -> tuple[list[float], list[float]]:
    """Return non-negative protrusion magnitude profile p(x) for scoring.

    Uses outward top edge profile, converts y<=0 to positive protrusion depth.
    """
    poly = edge_profile(
        length=length,
        outward=True,
        feature=True,
        template=template,
        tab_width=tab_width,
        tab_depth=tab_depth,
        ramp=ramp,
        asymmetry=asymmetry,
    )

    xs = _linspace(0.0, length, samples)
    ys = [0.0 for _ in xs]

    j = 0
    for i, x in enumerate(xs):
        while j + 1 < len(poly) and poly[j + 1][0] < x:
            j += 1
        x1, y1 = poly[j]
        x2, y2 = poly[min(j + 1, len(poly) - 1)]
        if abs(x2 - x1) < 1e-12:
            y = y2
        else:
            t = (x - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
        ys[i] = max(0.0, -y)
    return xs, ys


def shifted_value(xs: Sequence[float], ys: Sequence[float], qx: float) -> float:
    if qx <= xs[0] or qx >= xs[-1]:
        return 0.0
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] < qx:
            lo = mid
        else:
            hi = mid
    x1, x2 = xs[lo], xs[hi]
    y1, y2 = ys[lo], ys[hi]
    if abs(x2 - x1) < 1e-12:
        return y1
    t = (qx - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def required_separation_for_shift(
    xs: Sequence[float],
    protrusion: Sequence[float],
    shift: float,
) -> float:
    """Minimum normal gap required to avoid overlap at a tangential shift."""
    req = 0.0
    for x, p in zip(xs, protrusion):
        p_shift = shifted_value(xs, protrusion, x - shift)
        req = max(req, p_shift - p)
    return max(0.0, req)


def score_template(
    length: float,
    template: str,
    tab_width: float,
    tab_depth: float,
    ramp: float,
    asymmetry: float,
) -> Candidate:
    xs, p = profile_depth_function(length, template, tab_width, tab_depth, ramp, asymmetry)

    slide_shifts = [0.5, 1.0, 1.5, 2.0, 2.5]
    lock_vals = [required_separation_for_shift(xs, p, s) for s in slide_shifts]
    lock_strength = min(lock_vals)

    approach_gap = 0.95 * tab_depth
    admissible = []
    for s in _linspace(-3.0, 3.0, 81):
        req = required_separation_for_shift(xs, p, s)
        if req <= approach_gap:
            admissible.append(s)
    approach_window = (max(admissible) - min(admissible)) if admissible else 0.0

    # Prefer strong lock against 0.5-2.5 µm slide, but keep nontrivial approach corridor.
    penalty = max(0.0, 1.0 - approach_window / 3.0)
    score = 1.35 * lock_strength + 1.0 * min(approach_window, 3.2) - 0.35 * penalty

    return Candidate(
        template=template,
        tab_depth=tab_depth,
        tab_width=tab_width,
        ramp=ramp,
        asymmetry=asymmetry,
        lock_strength=lock_strength,
        approach_window=approach_window,
        score=score,
    )


def rotate_point(pt: tuple[float, float], quarter_turns: int) -> tuple[float, float]:
    x, y = pt
    q = quarter_turns % 4
    if q == 0:
        return (x, y)
    if q == 1:
        return (y, -x)
    if q == 2:
        return (-x, -y)
    return (-y, x)


def transform_edge(
    local_pts: Iterable[tuple[float, float]],
    quarter_turns: int,
    offset: tuple[float, float],
) -> list[tuple[float, float]]:
    out = []
    for pt in local_pts:
        xr, yr = rotate_point(pt, quarter_turns)
        out.append((xr + offset[0], yr + offset[1]))
    return out


def chamfer_polygon(points: list[tuple[float, float]], c: float) -> list[tuple[float, float]]:
    if c <= 0:
        return points
    chamfered: list[tuple[float, float]] = []
    n = len(points)
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p = points[i]
        p_next = points[(i + 1) % n]
        v1x, v1y = p_prev[0] - p[0], p_prev[1] - p[1]
        v2x, v2y = p_next[0] - p[0], p_next[1] - p[1]
        l1 = math.hypot(v1x, v1y)
        l2 = math.hypot(v2x, v2y)
        if l1 < 1e-9 or l2 < 1e-9:
            chamfered.append(p)
            continue
        c1 = min(c, 0.45 * l1)
        c2 = min(c, 0.45 * l2)
        a = (p[0] + v1x / l1 * c1, p[1] + v1y / l1 * c1)
        b = (p[0] + v2x / l2 * c2, p[1] + v2y / l2 * c2)
        chamfered.extend([a, b])
    return chamfered


def make_particle_polygon(
    spec: ParticleSpec,
    features: EdgeFeatures,
    polarity: str,
    template: str,
    ramp: float,
    asymmetry: float,
) -> list[tuple[float, float]]:
    W, H = spec.width, spec.height

    if polarity == "self":
        edge_outward = [True, False, True, False]
    elif polarity == "A":
        edge_outward = [True, True, False, False]
    elif polarity == "B":
        edge_outward = [False, False, True, True]
    else:
        raise ValueError(f"Unknown polarity: {polarity}")

    top_local = edge_profile(W, edge_outward[0], features.top, template, spec.tab_width, spec.tab_depth, ramp, asymmetry)
    right_local = edge_profile(H, edge_outward[1], features.right, template, spec.tab_width, spec.tab_depth, ramp, asymmetry)
    bottom_local = edge_profile(W, edge_outward[2], features.bottom, template, spec.tab_width, spec.tab_depth, ramp, asymmetry)
    left_local = edge_profile(H, edge_outward[3], features.left, template, spec.tab_width, spec.tab_depth, ramp, asymmetry)

    top = transform_edge(top_local, 0, (-W / 2, H / 2))
    right = transform_edge(right_local, 1, (W / 2, H / 2))
    bottom = transform_edge(bottom_local, 2, (W / 2, -H / 2))
    left = transform_edge(left_local, 3, (-W / 2, -H / 2))

    outline = top[:-1] + right[:-1] + bottom[:-1] + left[:-1]
    return chamfer_polygon(outline, spec.corner_chamfer)


def polygon_to_svg_path(points: list[tuple[float, float]]) -> str:
    cmds = [f"M {points[0][0]:.3f},{-points[0][1]:.3f}"]
    cmds.extend(f"L {x:.3f},{-y:.3f}" for x, y in points[1:])
    cmds.append("Z")
    return " ".join(cmds)


def write_svg(path: Path, polygons: list[tuple[str, list[tuple[float, float]], str]]) -> None:
    all_pts = [pt for _, poly, _ in polygons for pt in poly]
    xs, ys = [p[0] for p in all_pts], [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 3.0
    view_w = (max_x - min_x) + 2 * pad
    view_h = (max_y - min_y) + 2 * pad

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x-pad:.3f} {-max_y-pad:.3f} {view_w:.3f} {view_h:.3f}">',
        '<rect width="100%" height="100%" fill="#f8f8f8"/>',
        '<g stroke="#b22" stroke-width="0.14" fill="#ffd6d6" fill-opacity="0.75">',
    ]
    for label, poly, color in polygons:
        lines.append(f'<path d="{polygon_to_svg_path(poly)}" fill="{color}"/>')
        cx = sum(x for x, _ in poly) / len(poly)
        cy = sum(y for _, y in poly) / len(poly)
        lines.append(
            f'<text x="{cx:.3f}" y="{-cy:.3f}" text-anchor="middle" dominant-baseline="middle" '
            'font-size="1.6" fill="#222">'
            f'{label}</text>'
        )
    lines.extend(['</g>', '</svg>'])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_baseline(output_dir: Path, spec: ParticleSpec) -> None:
    y_only = EdgeFeatures(top=True, right=False, bottom=True, left=False)
    all_edges = EdgeFeatures(top=True, right=True, bottom=True, left=True)

    self_y = make_particle_polygon(spec, y_only, "self", "single_tooth", ramp=0.7, asymmetry=0.0)
    self_4 = make_particle_polygon(spec, all_edges, "self", "single_tooth", ramp=0.7, asymmetry=0.0)
    write_svg(output_dir / "single_shape_y_lock.svg", [("self-Y", self_y, "#ffd6d6")])
    write_svg(output_dir / "single_shape_4edge_lock.svg", [("self-4E", self_4, "#ffd6d6")])

    pairA_y = make_particle_polygon(spec, y_only, "A", "dovetail", ramp=0.8, asymmetry=0.0)
    pairB_y = make_particle_polygon(spec, y_only, "B", "dovetail", ramp=0.8, asymmetry=0.0)
    pairB_y = [(x + spec.width + 5.0, y) for x, y in pairB_y]
    write_svg(output_dir / "pair_y_lock.svg", [("A", pairA_y, "#ffd6d6"), ("B", pairB_y, "#d8e5ff")])

    pairA_4 = make_particle_polygon(spec, all_edges, "A", "double_tooth_asym", ramp=0.7, asymmetry=0.4)
    pairB_4 = make_particle_polygon(spec, all_edges, "B", "double_tooth_asym", ramp=0.7, asymmetry=0.4)
    pairB_4 = [(x + spec.width + 5.0, y) for x, y in pairB_4]
    write_svg(output_dir / "pair_4edge_lock.svg", [("A", pairA_4, "#ffd6d6"), ("B", pairB_4, "#d8e5ff")])


def run_search(output_dir: Path, spec: ParticleSpec, top_k: int) -> list[Candidate]:
    templates = ["single_tooth", "double_tooth_asym", "dovetail"]
    depths = [1.0, 1.2, 1.4, 1.6, 1.8]
    widths = [3.0, 3.5, 4.0, 4.5, 5.0]
    ramps = [0.5, 0.7, 0.9]
    asymmetries = [0.0, 0.2, 0.4, 0.6]

    candidates: list[Candidate] = []
    for template in templates:
        for d in depths:
            for w in widths:
                for r in ramps:
                    for a in asymmetries:
                        if template != "double_tooth_asym" and a > 0.0:
                            continue
                        c = score_template(spec.width, template, w, d, r, a)
                        candidates.append(c)

    ranked_all = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Keep diversity so the user gets a few distinct geometries.
    best_by_template: list[Candidate] = []
    seen_templates: set[str] = set()
    for c in ranked_all:
        if c.template in seen_templates:
            continue
        best_by_template.append(c)
        seen_templates.add(c.template)

    ranked: list[Candidate] = best_by_template[:]
    for c in ranked_all:
        if len(ranked) >= top_k:
            break
        if c in ranked:
            continue
        ranked.append(c)
    ranked = ranked[:top_k]

    y_only = EdgeFeatures(top=True, right=False, bottom=True, left=False)
    all_edges = EdgeFeatures(top=True, right=True, bottom=True, left=True)

    report = []
    for i, c in enumerate(ranked, start=1):
        local_spec = ParticleSpec(
            width=spec.width,
            height=spec.height,
            tab_depth=c.tab_depth,
            tab_width=c.tab_width,
            corner_chamfer=spec.corner_chamfer,
        )

        # Export as complementary 4-edge pair and self Y-lock to show two deployment modes.
        pairA = make_particle_polygon(local_spec, all_edges, "A", c.template, c.ramp, c.asymmetry)
        pairB = make_particle_polygon(local_spec, all_edges, "B", c.template, c.ramp, c.asymmetry)
        pairB = [(x + spec.width + 5.0, y) for x, y in pairB]
        write_svg(
            output_dir / f"candidate_{i:02d}_pair4.svg",
            [(f"A{i}", pairA, "#ffd6d6"), (f"B{i}", pairB, "#d8e5ff")],
        )

        selfy = make_particle_polygon(local_spec, y_only, "self", c.template, c.ramp, c.asymmetry)
        write_svg(output_dir / f"candidate_{i:02d}_selfY.svg", [(f"S{i}", selfy, "#ffd6d6")])

        report.append(
            {
                "rank": i,
                "template": c.template,
                "tab_depth": c.tab_depth,
                "tab_width": c.tab_width,
                "ramp": c.ramp,
                "asymmetry": c.asymmetry,
                "lock_strength_um": round(c.lock_strength, 3),
                "approach_window_um": round(c.approach_window, 3),
                "score": round(c.score, 3),
                "files": [
                    f"candidate_{i:02d}_pair4.svg",
                    f"candidate_{i:02d}_selfY.svg",
                ],
            }
        )

    (output_dir / "candidate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Candidate lock-and-key shapes",
        "",
        "Auto-searched candidates that balance anti-sliding lock strength and approach corridor.",
        "",
        "| Rank | Template | tab_depth (µm) | tab_width (µm) | ramp (µm) | asymmetry | lock_strength (µm) | approach_window (µm) | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report:
        md_lines.append(
            f"| {row['rank']} | {row['template']} | {row['tab_depth']} | {row['tab_width']} | {row['ramp']} | "
            f"{row['asymmetry']} | {row['lock_strength_um']} | {row['approach_window_um']} | {row['score']} |"
        )
    (output_dir / "candidate_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "search"], default="search")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--width", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--tab-depth", type=float, default=1.6)
    parser.add_argument("--tab-width", type=float, default=4.0)
    parser.add_argument("--corner-chamfer", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = ParticleSpec(
        width=args.width,
        height=args.height,
        tab_depth=args.tab_depth,
        tab_width=args.tab_width,
        corner_chamfer=args.corner_chamfer,
    )

    if args.mode == "baseline":
        export_baseline(output_dir, spec)
        print(f"Wrote baseline SVG files to: {output_dir.resolve()}")
        return

    ranked = run_search(output_dir, spec, top_k=args.top_k)
    print(f"Searched {len(ranked)} top candidates. Results in: {output_dir.resolve()}")
    for i, c in enumerate(ranked, start=1):
        print(
            f"#{i}: {c.template} depth={c.tab_depth:.2f} width={c.tab_width:.2f} "
            f"ramp={c.ramp:.2f} asym={c.asymmetry:.2f} "
            f"lock={c.lock_strength:.3f}µm approach={c.approach_window:.3f}µm score={c.score:.3f}"
        )


if __name__ == "__main__":
    main()

"""Inline-SVG charts for the dashboard's analytics lab.

Presentation only, like dashboard.py: every function takes plain dicts /
DataFrames that the analytics layer already computed and returns an SVG
string. No plotting library -- the page stays a single self-contained file
and every mark can carry a `data-tip` the dashboard's one tooltip handler
reads on hover.

Conventions (shared by every chart here):
  * line identity uses the official line colours from config, always paired
    with a direct text label (red/green fail CVD separation on their own);
  * values and labels wear text colours, never the series colour;
  * axes and grids are recessive hairlines; one y-axis per chart, always.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd

from src import config

INK, TEXT, MUTED, FAINT, RULE = "#0B0B0C", "#F2EFE6", "#A39F94", "#6B675F", "rgba(242,239,230,.10)"
PANEL = "#161617"
SIGNAL = "#FFC72C"


def _primary_color(line: str) -> str:
    for token in str(line).split("/"):
        if token in config.LINE_COLORS:
            return config.LINE_COLORS[token]
    return config.LINE_COLORS["Interchange"]


def indian_group(n: float) -> str:
    """12345678 -> '1,23,45,678' (lakh/crore grouping)."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def indian_short(n: float) -> str:
    """Compact Indian units: 1.4 Cr, 8.0 L, 45K."""
    if n >= 1e7:
        return f"{n / 1e7:.2f} Cr"
    if n >= 1e5:
        return f"{n / 1e5:.1f} L"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(int(n))


def _tip(*rows: str) -> str:
    return escape("|".join(rows), quote=True)


def _svg_open(w: int, h: int, label: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{escape(label, quote=True)}" '
            f'preserveAspectRatio="xMidYMid meet">')


# ---------------------------------------------------------------------------
# 1. Crowd vs. structure: betweenness against footfall (log x)
# ---------------------------------------------------------------------------
def scatter_crowd_vs_structure(scatter: list[dict]) -> str:
    W, H = 1080, 440
    L, R, T, B = 64, 24, 20, 48
    df = pd.DataFrame(scatter)
    x_min, x_max = 3e3, 1e6
    y_max = max(0.05, df["betweenness"].max() * 1.08)

    def sx(v):
        v = min(max(v, x_min), x_max)
        return L + (math.log10(v) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * (W - L - R)

    def sy(v):
        return T + (1 - v / y_max) * (H - T - B)

    out = [_svg_open(W, H, "Scatter of station betweenness centrality against daily footfall, log scale")]
    # grid + axes
    for v in (1e4, 1e5, 1e6):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="{RULE}"/>')
        out.append(f'<text x="{x:.1f}" y="{H - B + 20}" class="ax" text-anchor="middle">{indian_short(v)}</text>')
    steps = 4
    for i in range(steps + 1):
        v = y_max * i / steps
        y = sy(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{RULE}"/>')
        out.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="ax" text-anchor="end">{v:.2f}</text>')
    out.append(f'<text x="{W - R}" y="{H - 8}" class="ax-title" text-anchor="end">Daily footfall (log scale) &rarr;</text>')
    out.append(f'<text x="{L}" y="{T - 6}" class="ax-title">&uarr; Betweenness centrality</text>')

    # median guides -> four quadrants
    mx, my = df["footfall"].median(), df["betweenness"].median()
    out.append(f'<line x1="{sx(mx):.1f}" y1="{T}" x2="{sx(mx):.1f}" y2="{H - B}" stroke="{FAINT}" stroke-dasharray="3 5"/>')
    out.append(f'<line x1="{L}" y1="{sy(my):.1f}" x2="{W - R}" y2="{sy(my):.1f}" stroke="{FAINT}" stroke-dasharray="3 5"/>')
    out.append(f'<text x="{sx(mx) - 10:.1f}" y="{T + 16}" class="quad" text-anchor="end">HIDDEN CHOKEPOINTS</text>')
    out.append(f'<text x="{W - R - 6}" y="{T + 16}" class="quad" text-anchor="end">CROWDED &amp; CRITICAL</text>')
    out.append(f'<text x="{W - R - 6}" y="{H - B - 10}" class="quad" text-anchor="end">BUSY BUT BYPASSABLE</text>')

    # dots: small first so hubs draw on top
    for r in df.sort_values("betweenness").itertuples(index=False):
        x, y = sx(r.footfall), sy(r.betweenness)
        rad = 7 if r.hub else 5
        tip = _tip(r.name, r.line.replace("/", " · "), f"{indian_group(r.footfall)} riders/day",
                   f"betweenness {r.betweenness:.3f}")
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" fill="{_primary_color(r.line)}" '
                   f'stroke="{PANEL}" stroke-width="2" class="mark" data-tip="{tip}"/>')

    # selective direct labels: top betweenness + top crowd
    label_ids = set(df.nlargest(6, "betweenness")["id"]) | set(df.nlargest(3, "footfall")["id"])
    for r in df[df["id"].isin(label_ids)].itertuples(index=False):
        x, y = sx(r.footfall), sy(r.betweenness)
        anchor, dx = ("end", -11) if x > W - 160 else ("start", 11)
        out.append(f'<text x="{x + dx:.1f}" y="{y + 4:.1f}" class="dl" text-anchor="{anchor}">{escape(r.name)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. Lorenz curve of footfall
# ---------------------------------------------------------------------------
def lorenz_curve(lorenz: list[float], gini: float, top10_share: float) -> str:
    W, H = 520, 360
    L, R, T, B = 52, 20, 20, 44
    n = len(lorenz) - 1

    def sx(i):
        return L + i / n * (W - L - R)

    def sy(v):
        return T + (1 - v) * (H - T - B)

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(lorenz))
    out = [_svg_open(W, H, f"Lorenz curve of station footfall, Gini {gini}")]
    for f in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" y1="{sy(f):.1f}" x2="{W - R}" y2="{sy(f):.1f}" stroke="{RULE}"/>')
        out.append(f'<text x="{L - 8}" y="{sy(f) + 4:.1f}" class="ax" text-anchor="end">{int(f * 100)}%</text>')
        out.append(f'<text x="{sx(f * n):.1f}" y="{H - B + 18}" class="ax" text-anchor="middle">{int(f * 100)}%</text>')
    out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(n)}" y2="{sy(1)}" stroke="{MUTED}" stroke-dasharray="4 5"/>')
    out.append(f'<text x="{sx(n * .52):.1f}" y="{sy(.56):.1f}" class="dl muted" transform="rotate(-33 {sx(n * .52):.1f} {sy(.56):.1f})">perfect equality</text>')
    out.append(f'<polygon points="{sx(0)},{sy(0)} {pts} {sx(n)},{sy(0)}" fill="{SIGNAL}" fill-opacity=".10"/>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="{SIGNAL}" stroke-width="2.5" stroke-linejoin="round"/>')
    # top-10 marker
    i10 = n - 10
    v10 = lorenz[i10]
    out.append(f'<line x1="{sx(i10):.1f}" y1="{sy(v10):.1f}" x2="{sx(i10):.1f}" y2="{sy(1):.1f}" stroke="{TEXT}" stroke-width="1"/>')
    out.append(f'<circle cx="{sx(i10):.1f}" cy="{sy(v10):.1f}" r="5" fill="{TEXT}" stroke="{PANEL}" stroke-width="2" class="mark" '
               f'data-tip="{_tip("Busiest 10 stations", f"{top10_share * 100:.0f}% of all riders", f"{10 / n * 100:.0f}% of stations")}"/>')
    out.append(f'<text x="{sx(i10) - 8:.1f}" y="{sy((v10 + 1) / 2):.1f}" class="dl" text-anchor="end">top 10 &rarr; {top10_share * 100:.0f}% of riders</text>')
    out.append(f'<text x="{L}" y="{H - 6}" class="ax-title">Share of stations (quietest &rarr; busiest)</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 3. Reach from the central hub: cumulative share of stations vs minutes
# ---------------------------------------------------------------------------
def reach_curve(reach_mins: list[float], hub_name: str, within: dict) -> str:
    W, H = 520, 360
    L, R, T, B = 52, 20, 20, 44
    t_max = max(180, int(math.ceil(max(reach_mins) / 60.0)) * 60)
    n = len(reach_mins)

    def sx(t):
        return L + min(t, t_max) / t_max * (W - L - R)

    def sy(v):
        return T + (1 - v) * (H - T - B)

    out = [_svg_open(W, H, f"Cumulative share of stations reachable from {hub_name} by travel time")]
    for f in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" y1="{sy(f):.1f}" x2="{W - R}" y2="{sy(f):.1f}" stroke="{RULE}"/>')
        out.append(f'<text x="{L - 8}" y="{sy(f) + 4:.1f}" class="ax" text-anchor="end">{int(f * 100)}%</text>')
    for t in range(0, t_max + 1, 60):
        out.append(f'<text x="{sx(t):.1f}" y="{H - B + 18}" class="ax" text-anchor="middle">{t // 60}h</text>')
    # step line
    d = [f"M{sx(0):.1f},{sy(0):.1f}"]
    for i, t in enumerate(reach_mins, start=1):
        d.append(f"H{sx(t):.1f}V{sy(i / n):.1f}")
    d.append(f"H{sx(t_max):.1f}")
    out.append(f'<path d="{"".join(d)}" fill="none" stroke="{SIGNAL}" stroke-width="2.5"/>')
    for t in (30, 60, 90):
        v = within.get(t, within.get(str(t), 0))
        out.append(f'<line x1="{sx(t):.1f}" y1="{sy(0)}" x2="{sx(t):.1f}" y2="{sy(v):.1f}" stroke="{FAINT}" stroke-dasharray="2 4"/>')
        out.append(f'<circle cx="{sx(t):.1f}" cy="{sy(v):.1f}" r="5" fill="{TEXT}" stroke="{PANEL}" stroke-width="2" class="mark" '
                   f'data-tip="{_tip(f"Within {t} min of {hub_name}", f"{v * 100:.0f}% of stations")}"/>')
        out.append(f'<text x="{sx(t) + 8:.1f}" y="{sy(v) + 16:.1f}" class="dl">{v * 100:.0f}%</text>')
    out.append(f'<text x="{L}" y="{H - 6}" class="ax-title">Travel time from {escape(hub_name)} &rarr;</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 4. Histogram helper (circuity, coordinate audit)
# ---------------------------------------------------------------------------
def bar_histogram(labels: list[str], counts: list[int], tips: list[str], highlight_from: int | None = None,
                  aria: str = "") -> str:
    W, H = 520, 300
    L, R, T, B = 40, 12, 24, 52
    n = len(counts)
    c_max = max(max(counts), 1)
    slot = (W - L - R) / n
    bw = slot - 10

    def sy(v):
        return T + (1 - v / c_max) * (H - T - B)

    out = [_svg_open(W, H, aria)]
    out.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="{MUTED}"/>')
    for i, (lab, c, tip) in enumerate(zip(labels, counts, tips)):
        x = L + i * slot + 5
        y = sy(c)
        h = H - B - y
        fill = SIGNAL if (highlight_from is None or i < highlight_from) else "#8A8579"
        if highlight_from is not None and i >= highlight_from:
            fill = "#E8743B"
        r = min(4, h / 2) if h > 0 else 0
        # rounded data-end, square baseline
        out.append(
            f'<path d="M{x:.1f},{H - B} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} '
            f'Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{H - B} Z" fill="{fill}" class="mark" data-tip="{tip}"/>'
        )
        out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" class="dl" text-anchor="middle">{c}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{H - B + 18}" class="ax" text-anchor="middle">{escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def circuity_histogram(track_df: pd.DataFrame) -> tuple[str, dict]:
    on = track_df[track_df["on_track"]]
    pct = (on["circuity"] - 1) * 100
    bins = [0, 1, 3, 5, 10, 20, 1e9]
    labels = ["<1%", "1-3%", "3-5%", "5-10%", "10-20%", "20%+"]
    counts = pd.cut(pct, bins=bins, right=False, labels=labels).value_counts().reindex(labels).fillna(0).astype(int).tolist()
    tips = [_tip(f"Detour {lab}", f"{c} segments") for lab, c in zip(labels, counts)]
    svg = bar_histogram(labels, counts, tips, aria="Histogram of track detour versus straight line per segment")
    stats = {
        "median_pct": round(float(pct.median()), 1),
        "mean_pct": round(float(pct.mean()), 1),
        "share_over_5": round(float((pct >= 5).mean()), 3),
        "n_on_track": int(len(on)),
        "n_total": int(len(track_df)),
        "track_km": round(float(track_df["track_km"].sum()), 1),
    }
    return svg, stats


def coordinate_audit(snaps_df: pd.DataFrame) -> tuple[str, dict]:
    m = snaps_df["snap_m"].fillna(0)
    bins = [0, 50, 150, 500, 2000, 1e9]
    labels = ["<50 m", "50-150 m", "150-500 m", "0.5-2 km", ">2 km"]
    counts = pd.cut(m, bins=bins, right=False, labels=labels).value_counts().reindex(labels).fillna(0).astype(int).tolist()
    tips = [_tip(f"Offset {lab}", f"{c} stations") for lab, c in zip(labels, counts)]
    svg = bar_histogram(labels, counts, tips, highlight_from=2,
                        aria="Histogram of station coordinate offset from the nearest track")
    stats = {"over_150": int((m > 150).sum()), "under_50": int((m < 50).sum()), "n": int(len(m)),
             "median_m": int(np.median(m))}
    return svg, stats

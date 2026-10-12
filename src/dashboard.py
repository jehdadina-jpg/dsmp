"""Standalone analytics dashboard: a single self-contained HTML report.

This module is presentation-only -- it takes already-computed results
(DataFrames + plain dicts from data_loader / graph_analytics /
spatial_analysis) and renders them as one polished HTML page: a WebGL 3D
render of the network graph, an animated route comparison, a centrality
leaderboard, and headline network/coverage stats. It performs no analysis
of its own and has no networkx/geopandas import, on purpose -- swap the
analytics layer and this module does not need to change.

Visual concept: a "platform departure board" meets neon transit map --
amber LED ticker + VT323 terminal type (real Indian Railways departure
boards are dot-matrix/split-flap displays) layered with the network's real
line colors pushed to a neon register. Not part of the original brief's
file list; added because a Folium map alone has limited room for a
"report" reading experience (stats, rankings, route comparisons). Kept as
its own module so spatial_analysis.py stays focused on GIS.
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from src import config, dashboard_charts as charts

logger = config.get_logger(__name__)


def _fmt_int(n: int) -> str:
    # Indian digit grouping (8,00,000), which is how every Mumbai reader reads a count.
    return charts.indian_group(n)


# The mark: a CSMT-style pointed Gothic arch (the terminus the whole network
# grew out of) with three lines -- Western, Central, Harbour -- converging
# into one hub dot. Drawn inline so the page stays a single file.
LOGO_SVG = (
    '<svg viewBox="0 0 48 48" class="logo-mark" aria-hidden="true">'
    '<rect width="48" height="48" fill="#FFC72C"/>'
    '<path d="M9 45V25C9 15 17 9 24 5C31 9 39 15 39 25V45Z" fill="#0B0B0C"/>'
    '<path d="M17 45V31Q17 24 24 20" stroke="#22C55E" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
    '<path d="M24 45V20" stroke="#EF4444" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
    '<path d="M31 45V31Q31 24 24 20" stroke="#3B82F6" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
    '<circle cx="24" cy="18.5" r="3.6" fill="#FFC72C" stroke="#0B0B0C" stroke-width="1.5"/>'
    '</svg>'
)


def _nav_html(network_summary: dict, generated_at: str) -> str:
    links = [("map", "Map"), ("routes", "Routes"), ("planner", "Planner"),
             ("analytics", "Analytics"), ("hubs", "Hubs"), ("method", "Method")]
    items = "".join(f'<a href="#{i}">{t}</a>' for i, t in links)
    return f"""
    <header class="nav">
      <div class="container nav-inner">
        <a class="brand" href="#top" aria-label="Local Pulse home">
          {LOGO_SVG}
          <span class="brand-text"><strong>Local Pulse</strong><span lang="hi">लोकल पल्स</span></span>
        </a>
        <nav class="nav-links" aria-label="Sections">{items}</nav>
        <span class="nav-meta"><i></i>{network_summary['num_stations']} stations</span>
      </div>
    </header>"""


def _line_badges(line: str) -> str:
    tokens = line.split("/")
    spans = []
    for t in tokens:
        color = config.LINE_COLORS.get(t, config.LINE_COLORS["Interchange"])
        spans.append(f'<span class="badge" style="--c:{color}">{t}</span>')
    return "".join(spans)


def _stat_tile(label: str, value: str, sub: str = "", delay: int = 0) -> str:
    return f"""
    <div class="stat-tile" style="--d:{delay}ms">
      <div class="stat-value">{value}</div>
      <div class="stat-label">{label}</div>
      {f'<div class="stat-sub">{sub}</div>' if sub else ''}
    </div>"""


def _leaderboard_row(rank: int, row: pd.Series, max_betweenness: float) -> str:
    pct = 0 if max_betweenness == 0 else round(100 * row["betweenness_centrality"] / max_betweenness, 1)
    return f"""
    <div class="lb-row">
      <div class="lb-rank">{rank:02d}</div>
      <div class="lb-name">
        {row['name']}
        <div class="lb-badges">{_line_badges(row['line'])}</div>
      </div>
      <div class="lb-bar-track"><div class="lb-bar" data-width="{pct}"></div></div>
      <div class="lb-metrics">
        <span title="Betweenness centrality">{row['betweenness_centrality']:.3f}</span>
        <span class="muted"> / </span>
        <span title="Degree centrality">{row['degree_centrality']:.3f}</span>
      </div>
      <div class="lb-footfall">{_fmt_int(int(row['daily_footfall']))}<span class="muted">/day</span></div>
    </div>"""


def _route_svg(route: dict, stations_df: pd.DataFrame) -> str:
    """A horizontal SVG strip: station dots joined by a line, with a yellow
    'train' segment sweeping along it via SMIL -- no JS needed. Only the
    endpoints and interchanges are labelled (alternating above/below) so
    long routes stay legible; the strip scrolls rather than squashing."""
    ids = route["path_ids"]
    n = len(ids)
    width = max(560, 30 * (n - 1) + 120)
    y = 50
    xs = [60 + i * (width - 120) / max(n - 1, 1) for i in range(n)]
    label_flip = False
    line_by_id = stations_df.set_index("station_id")["line"]

    dots, labels = [], []
    for i, (sid, x) in enumerate(zip(ids, xs)):
        color = config.LINE_COLORS.get(line_by_id.get(sid, "").split("/")[0], "#9aa4b8")
        is_hub = "/" in str(line_by_id.get(sid, ""))
        r = 9 if is_hub else 6
        # Interchanges drawn the way printed rail maps do it: a white-ringed
        # "capsule" stop, so they read as hubs without any animation.
        if is_hub:
            dots.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#0B0B0C" stroke="#F2EFE6" stroke-width="3"/>'
                        f'<circle cx="{x}" cy="{y}" r="{r-5}" fill="{color}"/>')
        else:
            dots.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="#161617" stroke-width="2.5"/>')
        is_end = i in (0, n - 1)
        if is_end or is_hub:
            ly = y - 18 if label_flip else y + 28
            label_flip = not label_flip
            cls = "route-label end" if is_end else "route-label"
            labels.append(
                f'<text x="{x}" y="{ly}" text-anchor="middle" class="{cls}">{route["path_names"][i]}</text>'
            )

    path_d = f"M{xs[0]},{y} " + " ".join(f"L{x},{y}" for x in xs[1:])
    duration = max(2.4, min(10, n * 0.9))

    return f"""
    <svg viewBox="0 0 {width} 92" class="route-svg" style="width:{width}px" preserveAspectRatio="xMinYMid meet">
      <path d="{path_d}" stroke="rgba(242,239,230,0.22)" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="{path_d}" stroke="#FFC72C" stroke-width="4" fill="none" stroke-linecap="round" pathLength="1" stroke-dasharray="0.06 0.94">
        <animate attributeName="stroke-dashoffset" from="1" to="0" dur="{duration}s" repeatCount="indefinite"/>
      </path>
      {''.join(dots)}
      {''.join(labels)}
    </svg>"""


def _route_card(title: str, route: dict, stations_df: pd.DataFrame, beam: bool = False) -> str:
    inner = f"""
      <div class="route-card-head">
        <h3>{title}</h3>
        <div class="route-totals">
          <span><strong>{route['total_time_mins']:.0f}</strong> min</span>
          <span><strong>{route['total_distance_km']:.1f}</strong> km</span>
          <span><strong>{route['num_stations']}</strong> stops</span>
          <span><strong>{route['interchanges']}</strong> interchange{'s' if route['interchanges'] != 1 else ''}</span>
          <span><strong>&#8377;{route['fare_inr']}</strong> fare</span>
          <span><strong>{route['co2_saved_kg']:.1f}</strong> kg CO&#8322; saved</span>
        </div>
      </div>
      <div class="ticket-tear"></div>
      <div class="route-svg-wrap">{_route_svg(route, stations_df)}</div>"""
    return f'<div class="route-card">{inner}</div>'


_CSS = """
:root{
  --ink:#0B0B0C; --board:#111112; --panel:#161617; --panel-2:#1D1D1F;
  --rule:rgba(242,239,230,.09); --rule-strong:rgba(242,239,230,.18);
  --text:#F2EFE6; --muted:#A39F94; --faint:#6B675F;
  --signal:#FFC72C; --signal-ink:#141100;
  --western:#22C55E; --central:#EF4444; --harbour:#3B82F6; --transharbour:#F59E0B;
  --font-display:'Archivo',system-ui,sans-serif; --font-body:'Archivo',system-ui,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,monospace; --font-dev:'Noto Sans Devanagari',sans-serif;
  --ease:cubic-bezier(.16,1,.3,1);
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--ink); color:var(--text); font-family:var(--font-body); font-size:15px;
  line-height:1.55; -webkit-font-smoothing:antialiased; font-variation-settings:"wdth" 100;
}
::selection{ background:var(--signal); color:var(--signal-ink); }
.container{max-width:1200px; margin:0 auto; padding:0 32px;}
@media (max-width:640px){ .container{padding:0 16px;} }
a{color:var(--signal);}
:focus-visible{outline:2px solid var(--signal); outline-offset:3px;}
.num{font-family:var(--font-mono); font-variant-numeric:tabular-nums;}

/* ---------- Departure-board ticker ---------- */
.marquee-bar{max-width:100vw; 
  background:#000; border-bottom:1px solid var(--rule-strong); overflow:hidden; white-space:nowrap;
  display:flex; align-items:stretch;
}
.marquee-tag{
  flex:none; background:var(--signal); color:var(--signal-ink); font-family:var(--font-display);
  font-weight:800; font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  padding:9px 14px; display:flex; align-items:center; gap:8px; position:relative; z-index:2;
}
.marquee-tag::before{content:""; width:7px; height:7px; border-radius:50%; background:var(--signal-ink); animation:blink 1.4s steps(2) infinite;}
@keyframes blink{ 50%{ opacity:.15; } }
.marquee-viewport{overflow:hidden; flex:1; min-width:0; display:flex; align-items:center;}
.marquee-track{ display:inline-block; padding-left:100%; animation:marquee var(--marquee-dur,32s) linear infinite; }
.marquee-track span{
  font-family:var(--font-mono); font-size:12.5px; letter-spacing:.08em; color:var(--signal);
}
.marquee-track .sep{ color:var(--faint); margin:0 20px; }
@keyframes marquee{ to{ transform:translateX(-100%); } }

/* ---------- Hero: the station nameboard ---------- */
.hero{padding:48px 0 8px;}
.board{
  background:var(--signal); color:var(--signal-ink); padding:6px;
  box-shadow:0 30px 80px -30px rgba(255,199,44,.35);
}
.board-inner{
  border:3px solid var(--signal-ink); padding:22px 30px 24px;
  display:grid; grid-template-columns:1fr auto; gap:12px 32px; align-items:end;
}
.board-dev{font-family:var(--font-dev); font-weight:700; font-size:clamp(18px,2.4vw,26px); line-height:1.2; grid-column:1 / -1;}
h1.board-title{
  font-family:var(--font-display); font-weight:900; font-variation-settings:"wdth" 70;
  font-size:clamp(46px,9.2vw,124px); line-height:.86; letter-spacing:-.01em;
  text-transform:uppercase; margin:0;
}
.board-code{
  font-family:var(--font-mono); font-weight:500; font-size:12px; letter-spacing:.1em; text-align:right;
  text-transform:uppercase; line-height:1.7;
}
.board-code strong{display:block; font-family:var(--font-display); font-weight:900; font-size:44px; letter-spacing:0; line-height:1;}
.board-lines{display:flex; gap:6px; grid-column:1 / -1; margin-top:8px; flex-wrap:wrap;}
.board-line{
  font-family:var(--font-display); font-weight:700; font-size:11px; letter-spacing:.12em; text-transform:uppercase;
  background:var(--signal-ink); color:var(--signal); padding:5px 10px 5px 8px; display:inline-flex; align-items:center; gap:7px;
}
.board-line i{width:9px; height:9px; border-radius:50%; background:var(--c);}
@media (max-width:640px){
  .board-inner{grid-template-columns:1fr; padding:18px 18px 20px;}
  .board-code{text-align:left;}
}

.hero-grid{display:grid; grid-template-columns:1fr 1.05fr; gap:40px; align-items:center; padding-top:48px; padding-bottom:40px;}
@media (max-width:900px){ .hero-grid{grid-template-columns:1fr; gap:28px;} }
.eyebrow{
  font-family:var(--font-mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--signal); margin-bottom:16px;
}
.hero-headline{
  font-family:var(--font-display); font-weight:800; font-variation-settings:"wdth" 80;
  font-size:clamp(30px,4vw,46px); line-height:1.02; margin:0 0 18px; letter-spacing:-.01em;
}
.hero-headline em{font-style:normal; color:var(--signal);}
.lede{color:var(--muted); font-size:16px; max-width:50ch; margin:0;}
.hero-canvas-wrap{
  position:relative; height:380px; border:1px solid var(--rule); background:
  radial-gradient(circle at 50% 50%, #1A1913 0%, var(--board) 70%);
}
#graph3d{position:absolute; inset:0;}
.hero-canvas-wrap canvas{display:block; width:100%; height:100%;}
.hero-canvas-tag{
  position:absolute; left:0; bottom:0; font-family:var(--font-mono); font-size:10.5px; letter-spacing:.06em;
  color:var(--muted); background:var(--ink); padding:6px 10px; border-top:1px solid var(--rule); border-right:1px solid var(--rule);
}
.corner{position:absolute; width:14px; height:14px; border-color:var(--signal); border-style:solid;}
.corner.tl{top:-1px; left:-1px; border-width:2px 0 0 2px;} .corner.tr{top:-1px; right:-1px; border-width:2px 2px 0 0;}
.corner.br{bottom:-1px; right:-1px; border-width:0 2px 2px 0;}

/* ---------- Stat strip ---------- */
.stat-row{display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid var(--rule-strong); border-bottom:1px solid var(--rule);}
@media (max-width:760px){ .stat-row{grid-template-columns:repeat(2,1fr);} }
.stat-tile{padding:22px 22px 22px 0; animation:rise .7s var(--ease) backwards; animation-delay:var(--d,0ms);}
.stat-tile + .stat-tile{padding-left:22px; border-left:1px solid var(--rule);}
@media (max-width:760px){
  .stat-tile:nth-child(3){padding-left:0; border-left:none;}
  .stat-tile:nth-child(n+3){border-top:1px solid var(--rule);}
}
.stat-value{
  font-family:var(--font-display); font-weight:800; font-variation-settings:"wdth" 75;
  font-size:clamp(36px,4.6vw,56px); line-height:1; color:var(--text); font-variant-numeric:tabular-nums;
}
.stat-label{font-family:var(--font-mono); font-size:11px; color:var(--signal); margin-top:10px; text-transform:uppercase; letter-spacing:.1em;}
.stat-sub{font-size:13px; color:var(--muted); margin-top:3px;}
@keyframes rise{ from{ opacity:0; transform:translateY(12px);} to{ opacity:1; transform:none;} }

/* ---------- Sections (numbered like platforms) ---------- */
main{counter-reset:pf;}
section{padding:64px 0;}
section.numbered{counter-increment:pf;}
.section-head{display:grid; grid-template-columns:auto 1fr; gap:6px 20px; align-items:start; margin-bottom:28px;}
.pf{
  grid-row:span 2; width:58px; height:58px; background:var(--signal); color:var(--signal-ink);
  display:flex; flex-direction:column; align-items:center; justify-content:center; line-height:1;
}
.pf small{font-family:var(--font-mono); font-size:9px; letter-spacing:.12em;}
.pf small::after{content:" / फलाट"; font-family:var(--font-dev); letter-spacing:0;}
.pf::after{content:"०" counter(pf, devanagari); font-family:var(--font-dev); font-weight:800; font-size:26px; margin-top:2px;}
h2.section-title{
  font-family:var(--font-display); font-weight:800; font-variation-settings:"wdth" 80;
  font-size:clamp(26px,3.2vw,36px); line-height:1.05; margin:0; letter-spacing:-.01em;
}
.section-desc{color:var(--muted); font-size:15px; margin:0; max-width:64ch;}
.divider{height:6px; background:repeating-linear-gradient(-45deg,var(--signal) 0 8px,var(--ink) 8px 16px); opacity:.85;}

/* ---------- Map preview ---------- */
.map-preview{position:relative; border:1px solid var(--rule-strong); background:var(--board);}
.map-preview iframe{display:block; width:100%; height:560px; border:0;}
@media (max-width:760px){ .map-preview iframe{height:400px;} }
.map-preview-cta{
  position:absolute; bottom:52px; right:14px; z-index:5; background:var(--signal); color:var(--signal-ink);
  font-family:var(--font-display); font-weight:700; font-size:13px; letter-spacing:.04em;
  padding:10px 16px; text-decoration:none; display:inline-flex; align-items:center; gap:8px;
  transition:transform .2s var(--ease);
}
.map-preview-cta:hover{transform:translateY(-2px);}
.map-preview-note{
  border-top:1px solid var(--rule); padding:10px 14px; font-size:11px; letter-spacing:.06em;
  color:var(--faint); font-family:var(--font-mono); text-transform:uppercase;
}

/* ---------- How it works ---------- */
.how-grid{display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--rule);}
@media (max-width:900px){ .how-grid{grid-template-columns:1fr 1fr;} }
@media (max-width:560px){ .how-grid{grid-template-columns:1fr;} }
.how-step{padding:24px 22px; border-right:1px solid var(--rule); position:relative; display:flex; flex-direction:column;}
.how-step:last-child{border-right:none;}
@media (max-width:900px){ .how-step{border-bottom:1px solid var(--rule);} }
.how-step-num{font-family:var(--font-mono); font-size:11px; color:var(--signal); letter-spacing:.12em; margin-bottom:28px;}
.how-step h4{margin:0 0 8px; font-family:var(--font-display); font-weight:700; font-size:19px;}
.how-step p{margin:0 0 16px; font-size:14px; color:var(--muted); flex:1;}
.how-step code{
  display:block; font-family:var(--font-mono); font-size:11.5px; color:var(--text);
  background:var(--panel); border-left:2px solid var(--signal); padding:9px 11px; overflow-x:auto; white-space:nowrap;
}
.how-arrow{display:none;}

/* ---------- Route cards ---------- */
.route-grid{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px;}
@media (max-width:900px){ .route-grid{grid-template-columns:minmax(0,1fr);} }
.route-card{background:var(--panel); border:1px solid var(--rule); border-top:3px solid var(--signal); padding:22px 24px 18px;}
.route-card-head h3{
  font-family:var(--font-mono); font-weight:500; font-size:11px; margin:0 0 14px; color:var(--signal);
  letter-spacing:.12em; text-transform:uppercase;
}
.route-totals{display:grid; grid-template-columns:repeat(3,1fr); gap:14px 10px;}
.route-totals span{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; display:flex; flex-direction:column;}
.route-totals strong{
  color:var(--text); font-family:var(--font-display); font-weight:800; font-variation-settings:"wdth" 75;
  font-size:30px; letter-spacing:0; line-height:1.05; font-variant-numeric:tabular-nums; text-transform:none;
}
.ticket-tear{height:0; border-top:1px dashed var(--rule-strong); margin:18px -24px 10px; position:relative;}
.ticket-tear::before,.ticket-tear::after{
  content:""; position:absolute; top:-9px; width:16px; height:16px; border-radius:50%; background:var(--ink);
  border:1px solid var(--rule);
}
.ticket-tear::before{left:-9px;} .ticket-tear::after{right:-9px;}
.route-svg-wrap{overflow-x:auto;}
.route-svg{display:block; min-width:100%; height:92px;}
.route-label{font-size:11px; font-weight:600; fill:var(--muted); font-family:var(--font-body);}
.route-label.end{fill:var(--text); font-weight:800; font-size:12px;}
.route-svg-wrap{scrollbar-width:thin; scrollbar-color:var(--rule-strong) transparent;}

/* ---------- Leaderboard: departure-board table ---------- */
.leaderboard{border-top:1px solid var(--rule-strong);}
.lb-head,.lb-row{
  display:grid; grid-template-columns:52px 1.3fr 1.1fr 120px 130px; align-items:center; gap:18px;
}
.lb-head{
  font-family:var(--font-mono); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--faint); padding:12px 0; border-bottom:1px solid var(--rule);
}
.lb-head .r{text-align:right;}
.lb-row{padding:16px 0; border-bottom:1px solid var(--rule); transition:background .2s ease;}
.lb-row:hover{background:linear-gradient(90deg, rgba(255,199,44,.06), transparent 70%);}
.lb-rank{font-family:var(--font-mono); color:var(--signal); font-size:15px; letter-spacing:.06em;}
.lb-row:nth-child(2) .lb-rank,.lb-row:nth-child(3) .lb-rank,.lb-row:nth-child(4) .lb-rank{
  background:var(--signal); color:var(--signal-ink); width:36px; text-align:center; padding:4px 0; font-weight:500;
}
.lb-name{font-family:var(--font-display); font-weight:700; font-size:19px; font-variation-settings:"wdth" 85; line-height:1.15;}
.lb-badges{margin-top:6px; display:flex; gap:4px; flex-wrap:wrap;}
.badge{
  --c:#999; display:inline-flex; align-items:center; gap:5px; font-family:var(--font-body);
  font-size:10px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
}
.badge::before{content:""; width:14px; height:3px; background:var(--c);}
.lb-bar-track{height:6px; background:var(--panel-2);}
.lb-bar{height:100%; width:0; background:var(--signal); transition:width 1.2s var(--ease);}
.lb-metrics{font-family:var(--font-mono); font-size:12.5px; color:var(--text); text-align:right; font-variant-numeric:tabular-nums;}
.lb-metrics .muted{color:var(--faint);}
.lb-footfall{font-family:var(--font-mono); font-size:13px; color:var(--text); text-align:right; font-variant-numeric:tabular-nums;}
.lb-footfall .muted{color:var(--faint); font-size:10.5px;}
@media (max-width:760px){
  .lb-head{display:none;}
  .lb-row{grid-template-columns:44px 1fr auto; gap:6px 14px;}
  .lb-bar-track{grid-column:2 / -1; grid-row:2;}
  .lb-metrics{display:none;}
}

/* ---------- Coverage ---------- */
.coverage-grid{display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--rule-strong); border-bottom:1px solid var(--rule);}
@media (max-width:760px){
  .coverage-grid{grid-template-columns:1fr;}
  .coverage-grid .stat-tile + .stat-tile{padding-left:0; border-left:none; border-top:1px solid var(--rule);}
}
.cta-row{display:flex; gap:12px; margin-top:28px; flex-wrap:wrap;}
.btn{
  font-family:var(--font-display); font-weight:700; font-size:14px; letter-spacing:.03em; padding:13px 22px;
  text-decoration:none; display:inline-flex; align-items:center; gap:10px; cursor:pointer;
  transition:transform .2s var(--ease), background .2s ease, color .2s ease; border:1px solid transparent;
}
.btn-primary{background:var(--signal); color:var(--signal-ink);}
.btn-primary:hover{transform:translateY(-2px);}
.btn-ghost{background:transparent; border-color:var(--rule-strong); color:var(--text);}
.btn-ghost:hover{border-color:var(--signal); color:var(--signal);}

/* ---------- Live route planner ---------- */
.planner-card{background:var(--panel); border:1px solid var(--rule); padding:24px;}
.planner-row{display:grid; grid-template-columns:1fr auto 1fr 180px auto; gap:12px; align-items:end;}
@media (max-width:900px){ .planner-row{grid-template-columns:1fr auto 1fr;} .planner-row > :nth-child(n+4){grid-column:1 / -1;} }
@media (max-width:560px){ .planner-row{grid-template-columns:1fr;} .planner-swap{justify-self:start;} }
.planner-field label{
  display:block; font-family:var(--font-mono); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin-bottom:8px;
}
.planner-field input, .planner-field select{
  width:100%; height:48px; background:var(--ink); border:1px solid var(--rule-strong); color:var(--text);
  font-family:var(--font-display); font-weight:600; font-size:16px; padding:0 14px; border-radius:0;
  transition:border-color .2s ease;
}
.planner-field input::placeholder{color:var(--faint); font-weight:400;}
.planner-field input:focus, .planner-field select:focus{outline:none; border-color:var(--signal);}
.planner-swap{
  background:var(--ink); border:1px solid var(--rule-strong); color:var(--signal);
  width:48px; height:48px; cursor:pointer; font-size:18px; line-height:1; transition:border-color .2s ease, transform .3s var(--ease);
}
.planner-swap:hover{border-color:var(--signal); transform:rotate(180deg);}
#plannerGo{height:48px; justify-content:center;}
.planner-error{color:var(--central); font-size:14px; margin-top:14px; display:none;}
.planner-error.visible{display:block;}
.planner-result{margin-top:22px;}
.planner-result:empty{margin-top:0;}
.planner-result .route-card{background:var(--ink); animation:rise .5s var(--ease);}
.stop-chip-row{display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:14px;}
.stop-chip{
  display:inline-flex; align-items:center; gap:7px; font-family:var(--font-body); font-weight:600; font-size:13px;
  color:var(--text) !important; background:var(--panel); border:1px solid var(--rule); padding:5px 10px 5px 8px;
}
.stop-chip .dot{width:9px; height:9px; border-radius:50%; flex:none;}
.stop-chip.hub{border-color:var(--signal);}
.stop-arrow{color:var(--faint); font-size:12px;}

/* ---------- Footer (minimal) ---------- */
footer{border-top:1px solid var(--rule); padding:20px 0 28px; color:var(--faint); font-size:12px; font-family:var(--font-mono);}
.footer-row{display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;}

.fade-in{opacity:0; transform:translateY(16px); transition:opacity .8s var(--ease), transform .8s var(--ease);}
.fade-in.visible{opacity:1; transform:none;}


/* ---------- Nav + logo ---------- */
.nav{position:sticky; top:0; z-index:50; background:rgba(11,11,12,.86); backdrop-filter:blur(12px) saturate(1.2);
  border-bottom:1px solid var(--rule);}
.nav-inner{display:flex; align-items:center; gap:28px; height:64px;}
.brand{display:flex; align-items:center; gap:12px; text-decoration:none; color:var(--text); flex:none;}
.logo-mark{width:38px; height:38px; display:block;}
.brand-text{display:flex; flex-direction:column; line-height:1;}
.brand-text strong{font-family:var(--font-display); font-weight:900; font-variation-settings:"wdth" 72; font-size:22px; text-transform:uppercase; letter-spacing:.01em;}
.brand-text span{font-family:var(--font-dev); font-weight:700; font-size:11px; color:var(--signal); margin-top:3px;}
.nav-links{display:flex; gap:4px; margin-left:auto;}
.nav-links a{color:var(--muted); text-decoration:none; font-size:13px; font-weight:600; padding:8px 12px; transition:color .2s ease, background .2s ease;}
.nav-links a:hover{color:var(--ink); background:var(--signal);}
.nav-meta{font-family:var(--font-mono); font-size:11px; color:var(--faint); display:flex; align-items:center; gap:8px; flex:none;}
.nav-meta i{width:7px; height:7px; border-radius:50%; background:var(--western); box-shadow:0 0 0 3px rgba(34,197,94,.18);}
@media (max-width:980px){ .nav-meta{display:none;} }
@media (max-width:760px){ .nav-links{display:none;} .nav-inner{height:56px;} }
html{scroll-behavior:smooth; scroll-padding-top:72px;}

/* ---------- Jaali lattice behind the hero (Indian carved-screen motif) ---------- */
.hero{position:relative;}
.hero::before{
  content:""; position:absolute; inset:0; pointer-events:none; opacity:.07;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'%3E%3Cg fill='none' stroke='%23FFC72C' stroke-width='1.2'%3E%3Cpath d='M28 4 L52 28 L28 52 L4 28Z'/%3E%3Ccircle cx='28' cy='28' r='9'/%3E%3Cpath d='M28 0v8M28 48v8M0 28h8M48 28h8'/%3E%3C/g%3E%3C/svg%3E");
  -webkit-mask-image:linear-gradient(180deg,#000 0%,transparent 75%); mask-image:linear-gradient(180deg,#000 0%,transparent 75%);
}
.hero > *{position:relative;}

/* ---------- Kickers ---------- */
.kicker{font-family:var(--font-mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:6px;}
.kicker span{font-family:var(--font-dev); font-size:15px; letter-spacing:0; color:var(--signal); text-transform:none; font-weight:700;}

/* ---------- Analytics lab ---------- */
.kpi-row{margin-bottom:28px;}
.chart-card{background:var(--panel); border:1px solid var(--rule); padding:24px 24px 18px; margin-bottom:20px; min-width:0;}
.chart-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px;}
.chart-grid .chart-card{margin-bottom:0;}
@media (max-width:900px){ .chart-grid{grid-template-columns:minmax(0,1fr);} }
.chart-kicker{font-family:var(--font-mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--signal); display:flex; gap:10px; align-items:center; margin-bottom:10px;}
.chart-kicker span{background:var(--signal); color:var(--signal-ink); padding:2px 6px; font-weight:500;}
.chart-headline{font-family:var(--font-display); font-weight:800; font-variation-settings:"wdth" 82; font-size:clamp(20px,2.2vw,26px); line-height:1.15; margin:0 0 16px; max-width:34ch;}
.chart-card.wide .chart-headline{max-width:52ch;}
.chart-method{font-size:12.5px; color:var(--faint); margin:12px 0 0; max-width:80ch; line-height:1.55;}
.chart{display:block; width:100%; height:auto; overflow:visible;}
.chart-scroll{overflow-x:auto;} .chart-scroll .chart{min-width:760px;}
.chart .ax{font:500 11px var(--font-mono); fill:var(--faint);}
.chart .ax-title{font:500 10.5px var(--font-mono); fill:var(--muted); letter-spacing:.06em; text-transform:uppercase;}
.chart .dl{font:600 12px var(--font-body); fill:var(--text);}
.chart .dl.muted{fill:var(--faint); font-weight:500;}
.chart .quad{font:500 10px var(--font-mono); fill:var(--faint); letter-spacing:.12em;}
.chart .mark{cursor:crosshair; transition:opacity .15s ease;}
.chart:hover .mark{opacity:.55;} .chart .mark:hover{opacity:1;}
.legend{display:flex; flex-wrap:wrap; gap:6px 18px; margin:-4px 0 10px;}
.legend-item{display:inline-flex; align-items:center; gap:7px; font-size:12px; color:var(--muted); font-weight:600;}
.legend-item i{width:10px; height:10px; border-radius:50%;}
.legend-item i.ring{width:13px; height:13px; background:transparent; border:2px solid var(--muted);}
.resilience{border-top:3px solid #E8743B;}
.res-grid{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.1fr); gap:32px; align-items:end;}
@media (max-width:900px){ .res-grid{grid-template-columns:minmax(0,1fr);} }
.res-big{font-family:var(--font-display); font-weight:900; font-variation-settings:"wdth" 70; font-size:clamp(72px,10vw,120px); line-height:.85; color:var(--text);}
.res-big small{font-size:.32em; color:var(--faint); font-weight:700; margin-left:8px;}
.res-label{font-family:var(--font-mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);}
.stop-chip em{font-style:normal; font-family:var(--font-mono); font-size:11px; color:var(--faint); margin-left:2px;}

/* ---------- Tooltip ---------- */
.tip{position:fixed; z-index:100; pointer-events:none; background:var(--signal); color:var(--signal-ink);
  padding:8px 11px; font-size:12px; line-height:1.4; max-width:260px; opacity:0; transform:translateY(4px);
  transition:opacity .12s ease, transform .12s ease; box-shadow:0 10px 30px rgba(0,0,0,.45);}
.tip.on{opacity:1; transform:none;}
.tip b{display:block; font-family:var(--font-display); font-weight:800; font-size:14px;}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important;}
  .marquee-track{animation:none; padding-left:16px;}
  .fade-in{opacity:1; transform:none;}
}
"""


def _marquee_html(stations_df: pd.DataFrame) -> str:
    items = []
    for row in stations_df.sort_values("daily_footfall", ascending=False).itertuples(index=False):
        tag = " · INTERCHANGE" if row.interchange_hub else ""
        items.append(f'<span>{row.name.upper()}{tag}</span><span class="sep">&bull;</span>')
    strip = "".join(items)
    # A per-station-count guess for the duration was still not slow enough
    # in practice (font-width varies per name, so pixel distance doesn't
    # scale cleanly with item count). The reliable fix is to measure the
    # strip's actual rendered width at runtime and derive the duration from
    # a fixed, comfortable reading speed -- see the script at the bottom of
    # the page that sets --marquee-dur once the DOM has laid out.
    return f"""
    <div class="marquee-bar" aria-hidden="true">
      <div class="marquee-tag">Footfall board</div>
      <div class="marquee-viewport">
        <div class="marquee-track" id="marqueeTrack" style="--marquee-dur:90s">{strip}{strip}</div>
      </div>
    </div>"""


def _hero_html(stations_df: pd.DataFrame, network_summary: dict) -> str:
    board_lines = "".join(
        f'<span class="board-line" style="--c:{config.LINE_COLORS[name]}"><i></i>{name}</span>'
        for name in ("Western", "Central", "Harbour", "Trans-Harbour")
    )
    return f"""
    <section class="hero" id="top">
      <div class="container">
        <div class="board">
          <div class="board-inner">
            <div class="board-dev" lang="mr">मुंबई उपनगरीय रेल्वे</div>
            <h1 class="board-title">Mumbai<br>Suburban Rail</h1>
            <div class="board-code">Network code<strong>MSR</strong>{len(stations_df)} stations modeled</div>
            <div class="board-lines">{board_lines}</div>
          </div>
        </div>
      </div>
      <div class="container hero-grid">
        <div>
          <div class="eyebrow">Mumbai local, by the numbers</div>
          <h2 class="hero-headline">Which stations hold the whole network together, and how long it <em>really</em> takes to cross the city.</h2>
          <p class="lede">
            We mapped the Western, Central, Harbour and Trans-Harbour lines as a graph,
            laid them on the actual track, and asked the questions every daily commuter
            already has an opinion on. Here is what the data says.
          </p>
        </div>
        <div class="hero-canvas-wrap">
          <span class="corner tl"></span><span class="corner tr"></span><span class="corner br"></span>
          <div id="graph3d"></div>
          <div class="hero-canvas-tag">Force-directed graph &middot; drag to rotate</div>
        </div>
      </div>
      <div class="container stat-row">
        {_stat_tile("Stations modeled", str(network_summary['num_stations']), delay=0)}
        {_stat_tile("Track segments", str(network_summary['num_edges']), f"{network_summary['total_track_km']} km total", delay=80)}
        {_stat_tile("Daily footfall", charts.indian_short(network_summary['total_daily_footfall']), f"{_fmt_int(network_summary['total_daily_footfall'])} station entries a day", delay=160)}
        {_stat_tile("Interchange hubs", str(len(network_summary['interchange_hubs'])), "picked out by the graph", delay=240)}
      </div>
    </section>"""


def _map_preview_html() -> str:
    """A live, embedded preview of the Folium map right on the homepage --
    the map used to be reachable only via a small text link buried in the
    Coverage section at the bottom of the page; landing on it immediately
    gives the report an actual visual centerpiece instead of stats-first."""
    return f"""
    <section class="fade-in numbered" id="map">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">नकाशा</span> &middot; Map</div>
            <h2 class="section-title">The network on a map</h2>
            <p class="section-desc">Every line drawn on its real track. Zoom in, switch lines on and off,
              and tap a station to see how busy it is.</p>
          </div>
        </div>
        <div class="map-preview">
          <a class="map-preview-cta" href="maps/mumbai_rail_map.html" target="_blank" rel="noopener">Open full map &#8599;</a>
          <iframe src="maps/mumbai_rail_map.html" loading="lazy" title="Mumbai Suburban Rail interactive map"></iframe>
          <div class="map-preview-note">Track geometry from OpenStreetMap &middot; Leaflet</div>
        </div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _how_it_works_html() -> str:
    """Show the actual computation pipeline, not just its output -- each
    step names the real function/formula behind it so the numbers on this
    page read as computed, not asserted."""
    steps = [
        ("01", "Clean the data", "pandas checks every station and edge, drops duplicates "
         "and catches bad coordinates.", "load_and_clean_data()"),
        ("02", "Measure real distance", "Haversine great-circle formula over each edge's "
         "WGS84 coordinates, so the Earth's curve is accounted for.", "2 &middot; r &middot; asin(&radic;a)"),
        ("03", "Route with Dijkstra", "networkx finds the shortest path by travel time or "
         "distance across the full graph.", "nx.shortest_path(G, src, dst, weight=w)"),
        ("04", "Fare + CO&#8322; impact", "Distance maps to a published 2nd-class fare slab and "
         "a train-vs-car emissions delta.", "(121 - 41) g/km &times; distance"),
    ]
    cells = []
    for i, (num, title, desc, code) in enumerate(steps):
        if i > 0:
            cells.append('<div class="how-arrow">&rarr;</div>')
        cells.append(f"""
        <div class="how-step">
          <div class="how-step-num">STEP {num}</div>
          <h4>{title}</h4>
          <p>{desc}</p>
          <code>{code}</code>
        </div>""")
    return f"""
    <section class="fade-in numbered" id="method">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">गणित</span> &middot; Method</div>
            <h2 class="section-title">How we got these numbers</h2>
            <p class="section-desc">Four steps, all in the Python pipeline. Re-run it and every number on this page updates.</p>
          </div>
        </div>
        <div class="how-grid">{''.join(cells)}</div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _routes_html(routes: dict[str, dict], stations_df: pd.DataFrame) -> str:
    time_route, dist_route = routes["time"], routes["distance"]
    same = time_route["path_ids"] == dist_route["path_ids"]
    note = (
        "Both pick the same route this time."
        if same else
        "The shortest route and the fastest one change trains at different stations."
    )
    origin_name = time_route["path_names"][0]
    dest_name = time_route["path_names"][-1]
    return f"""
    <section class="fade-in numbered" id="routes">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">मार्ग</span> &middot; Route finder</div>
            <h2 class="section-title">{origin_name} &rarr; {dest_name}</h2>
            <p class="section-desc">One run optimised for minutes, one for kilometres. {note}</p>
          </div>
        </div>
        <div class="route-grid">
          {_route_card("Fastest by travel time", time_route, stations_df, beam=True)}
          {_route_card("Shortest by distance", dist_route, stations_df, beam=True)}
        </div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _planner_html(stations_df: pd.DataFrame) -> str:
    """Live, client-side route planner: pick *any* two stations in the
    browser and get an instant Dijkstra route -- unlike the two route cards
    above (baked in at build time for one fixed origin/destination pair),
    this runs entirely in JS against the embedded network graph, no Python
    re-run needed."""
    options = "".join(
        f'<option value="{name}"></option>'
        for name in sorted(stations_df["name"].unique())
    )
    return f"""
    <section class="fade-in numbered" id="planner">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">प्रवास</span> &middot; Journey planner</div>
            <h2 class="section-title">Plan your own trip</h2>
            <p class="section-desc">
              Pick any two of the {len(stations_df)} stations. The route is worked out right here in your browser.
            </p>
          </div>
        </div>
        <div class="planner-card">
          <div class="planner-row">
            <div class="planner-field">
              <label for="plannerFrom">From</label>
              <input id="plannerFrom" list="stationNames" placeholder="e.g. Andheri" autocomplete="off">
            </div>
            <button type="button" class="planner-swap" id="plannerSwap" title="Swap origin/destination">&#8646;</button>
            <div class="planner-field">
              <label for="plannerTo">To</label>
              <input id="plannerTo" list="stationNames" placeholder="e.g. Thane" autocomplete="off">
            </div>
            <div class="planner-field">
              <label for="plannerWeight">Optimize by</label>
              <select id="plannerWeight">
                <option value="time">Travel time</option>
                <option value="distance">Distance</option>
              </select>
            </div>
            <button type="button" class="btn btn-primary" id="plannerGo">Find route &rarr;</button>
          </div>
          <datalist id="stationNames">{options}</datalist>
          <div class="planner-error" id="plannerError"></div>
          <div class="planner-result" id="plannerResult"></div>
        </div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _leaderboard_html(centrality_df: pd.DataFrame, top_n: int = 8) -> str:
    top = centrality_df.head(top_n)
    max_b = centrality_df["betweenness_centrality"].max() or 1.0
    rows = "".join(_leaderboard_row(int(r["rank"]), r, max_b) for _, r in top.iterrows())
    return f"""
    <section class="fade-in numbered" id="hubs">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">जंक्शन</span> &middot; Junction ranking</div>
            <h2 class="section-title">The stations everything runs through</h2>
            <p class="section-desc">
              Ranked by betweenness: the share of all shortest trips that have to pass through a station.
              Nobody told the model about Kurla or Dadar. It found them.
            </p>
          </div>
        </div>
        <div class="leaderboard">
          <div class="lb-head"><span>Rank</span><span>Station</span><span>Betweenness</span><span class="r">Btw / Deg</span><span class="r">Footfall</span></div>
          {rows}
        </div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _coverage_html(buffer_stats: dict, network_summary: dict) -> str:
    n_hubs = len(network_summary["interchange_hubs"])
    return f"""
    <section class="fade-in numbered" id="coverage">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">पोहोच</span> &middot; Walkable reach</div>
            <h2 class="section-title">How far a hub reaches on foot</h2>
            <p class="section-desc">A 1 km walk around every interchange, roughly 12 to 15 minutes at a normal pace.</p>
          </div>
        </div>
        <div class="coverage-grid">
          {_stat_tile("Interchange hubs buffered", str(n_hubs))}
          {_stat_tile("Total catchment area", f"{buffer_stats.get('total_buffer_area_km2', 0):.1f} km²", "non-overlapping, 1 km radius each")}
          {_stat_tile("Avg. node degree", f"{network_summary.get('avg_degree', 0):.2f}", "mean connections per station")}
        </div>
        <div class="cta-row">
          <a class="btn btn-primary" href="maps/mumbai_rail_map.html">Open interactive map &rarr;</a>
          <a class="btn btn-ghost" href="../data/processed/interchange_buffers.geojson">Download buffers.geojson</a>
        </div>
      </div>
    </section>"""


def _chart_card(num: str, kicker: str, headline: str, method: str, body: str, extra_cls: str = "") -> str:
    return f"""
        <article class="chart-card {extra_cls}">
          <div class="chart-kicker"><span>{num}</span>{kicker}</div>
          <h3 class="chart-headline">{headline}</h3>
          {body}
          <p class="chart-method">{method}</p>
        </article>"""


def _analytics_html(insights: dict, spatial: dict) -> str:
    """The data-science section: every headline sentence below is computed
    from the numbers it sits on, never hand-written."""
    track_df, snaps_df = spatial["track_df"], spatial["snaps_df"]
    circ_svg, circ = charts.circuity_histogram(track_df)
    audit_svg, audit = charts.coordinate_audit(snaps_df)

    sc = pd.DataFrame(insights["scatter"])
    critical = sc[sc["betweenness"] >= sc["betweenness"].quantile(0.75)]
    hidden = critical.nsmallest(1, "footfall").iloc[0]
    within = insights["reach_within"]
    diam = insights["diameter"]
    cuts = insights["articulation_points"]
    n_st = len(sc)

    legend = "".join(
        f'<span class="legend-item"><i style="background:{config.LINE_COLORS[k]}"></i>{k}</span>'
        for k in ("Western", "Central", "Harbour", "Trans-Harbour")
    ) + '<span class="legend-item"><i class="ring"></i>Interchange (larger dot)</span>'

    kpis = "".join([
        _stat_tile("Footfall Gini", f"{insights['gini']:.2f}", "0 is perfectly even, 1 is one station taking everyone"),
        _stat_tile("Global efficiency", f"{insights['global_efficiency']:.3f}", "how directly stations connect, from 0 to 1"),
        _stat_tile("Mean trip", f"{insights['mean_trip_mins']:.0f} min", "average quickest trip between any two stations"),
        _stat_tile("Longest shortest trip", f"{diam['mins'] / 60:.1f} h", f"{diam['from']} &rarr; {diam['to']}"),
    ])

    scatter = _chart_card(
        "01", "Crowd vs. structure",
        f"{hidden['name']} routes {hidden['betweenness'] * 100:.0f}% of shortest paths on only "
        f"{charts.indian_short(hidden['footfall'])} riders a day.",
        "Each dot is a station: how busy it is against how many shortest trips pass through it. "
        "Dashed lines are the medians. Top left are the stations the network leans on more than their crowds suggest.",
        f'<div class="legend">{legend}</div><div class="chart-scroll">{charts.scatter_crowd_vs_structure(insights["scatter"])}</div>',
        "wide",
    )
    lorenz = _chart_card(
        "02", "Inequality",
        f"The busiest 10 stations handle {insights['top10_share'] * 100:.0f}% of all riders.",
        f"Lorenz curve of daily footfall, Gini {insights['gini']:.2f}. The further the curve sags below the diagonal, "
        "the more the crowd piles into a few stations.",
        charts.lorenz_curve(insights["lorenz"], insights["gini"], insights["top10_share"]),
    )
    reach = _chart_card(
        "03", "Reach",
        f"From {insights['hub_name']}, {within[60] * 100:.0f}% of stations are under an hour away.",
        f"Share of stations you can reach from {insights['hub_name']}, the top-ranked hub, as the minutes go up. "
        "Each step is one station.",
        charts.reach_curve(insights["reach_mins"], insights["hub_name"], within),
    )
    circuity = _chart_card(
        "04", "Track vs. straight line",
        f"Real track runs a median {circ['median_pct']:.1f}% longer than the straight line.",
        f"We traced {circ['n_on_track']} of {circ['n_total']} segments along OpenStreetMap track and compared each "
        f"with a straight line. Average detour {circ['mean_pct']:.1f}%, and {circ['share_over_5'] * 100:.0f}% of "
        "segments detour more than 5%.",
        circ_svg,
    )
    audit_card = _chart_card(
        "05", "Data audit",
        f"{audit['over_150']} of {audit['n']} source coordinates sit more than 150 m off the track.",
        f"How far each recorded station location sits from the nearest track. {audit['under_50']} are within 50 m. "
        "The orange ones are moved onto the track on the map, and their popups say so.",
        audit_svg,
    )
    top_cuts = "".join(
        f'<span class="stop-chip"><span class="dot" style="background:{charts._primary_color(c["line"])}"></span>'
        f'{c["name"]}<em>{charts.indian_short(c["footfall"])}</em></span>'
        for c in cuts[:8]
    )
    resilience = f"""
        <article class="chart-card resilience">
          <div class="chart-kicker"><span>06</span>Resilience</div>
          <div class="res-grid">
            <div>
              <div class="res-big">{len(cuts)}<small>/ {n_st}</small></div>
              <h3 class="chart-headline">stations are single points of failure.</h3>
              <p class="chart-method">Shut any one of these and the network splits in two. Only
                {insights['loop_station_count']} stations have another way around them, because the system is mostly
                long corridors running side by side.</p>
            </div>
            <div>
              <div class="res-label">Busiest single points of failure</div>
              <div class="stop-chip-row">{top_cuts}</div>
            </div>
          </div>
        </article>"""

    return f"""
    <section class="fade-in numbered analytics" id="analytics">
      <div class="container">
        <div class="section-head">
          <div class="pf" aria-hidden="true"><small>PF</small></div>
          <div>
            <div class="kicker"><span lang="mr">विश्लेषण</span> &middot; Analytics lab</div>
            <h2 class="section-title">What the data says</h2>
            <p class="section-desc">Six things we learned from the network. Hover over any point for the exact number.</p>
          </div>
        </div>
        <div class="stat-row kpi-row">{kpis}</div>
        {scatter}
        <div class="chart-grid">{lorenz}{reach}{circuity}{audit_card}</div>
        {resilience}
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _footer_html(generated_at: str) -> str:
    # Deliberately minimal: sources and caveats live in the README.
    return f"""
    <footer>
      <div class="container footer-row">
        <span>Data: Wikipedia &middot; OpenStreetMap &middot; see README for sources</span>
      </div>
    </footer>"""

_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
  // ---- departure-board marquee: derive speed from actual pixel width ----
  // A guessed duration (by station count, or a flat constant) drifts wrong
  // as soon as station names get longer/shorter or the count changes, the
  // only reliable fix is to measure the rendered strip and pick a duration
  // for a fixed, comfortable px/s reading speed once the browser has laid
  // it out.
  var track = document.getElementById('marqueeTrack');
  if (track) {
    var setMarqueeSpeed = function () {
      var stripWidth = track.scrollWidth / 2; // track = strip + strip
      var PX_PER_SEC = 100; // typical news-ticker pace, comfortably readable
      var duration = Math.max(30, Math.round(stripWidth / PX_PER_SEC));
      track.style.setProperty('--marquee-dur', duration + 's');
    };
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(setMarqueeSpeed);
    } else {
      setMarqueeSpeed();
    }
    window.addEventListener('resize', setMarqueeSpeed);
  }

  // ---- live route planner: Dijkstra over the embedded network, in-browser ----
  (function () {
    var NET = NETWORK_DATA_JSON;
    var LINE_COLORS = LINE_COLORS_JSON;

    // Build an adjacency list once: id -> [{to, line, dist, time}, ...].
    // The network is undirected (trains run both directions on the same
    // track), so every edge is added in both directions.
    var adj = {};
    Object.keys(NET.stations).forEach(function (id) { adj[id] = []; });
    NET.edges.forEach(function (e) {
      var a = e[0], b = e[1], line = e[2], dist = e[3], time = e[4];
      adj[a].push({ to: b, line: line, dist: dist, time: time });
      adj[b].push({ to: a, line: line, dist: dist, time: time });
    });

    var nameToId = {};
    Object.keys(NET.stations).forEach(function (id) {
      nameToId[NET.stations[id].n.toLowerCase()] = id;
    });

    function resolveStation(token) {
      if (!token) return null;
      var t = token.trim();
      if (NET.stations[t.toUpperCase()]) return t.toUpperCase();
      return nameToId[t.toLowerCase()] || null;
    }

    // Plain O(V^2) Dijkstra: the network has ~100-150 nodes, so a heap
    // buys nothing here and this stays trivially easy to read/verify.
    function dijkstra(srcId, dstId, weightKey) {
      var dist = {}, prev = {}, viaLine = {}, visited = {};
      Object.keys(NET.stations).forEach(function (id) { dist[id] = Infinity; });
      dist[srcId] = 0;

      while (true) {
        var u = null, best = Infinity;
        for (var id in dist) {
          if (!visited[id] && dist[id] < best) { best = dist[id]; u = id; }
        }
        if (u === null || u === dstId) break;
        visited[u] = true;
        adj[u].forEach(function (edge) {
          var w = weightKey === 'distance' ? edge.dist : edge.time;
          var alt = dist[u] + w;
          if (alt < dist[edge.to]) {
            dist[edge.to] = alt;
            prev[edge.to] = u;
            viaLine[edge.to] = edge;
          }
        });
      }
      if (dist[dstId] === undefined || dist[dstId] === Infinity) return null;

      var pathIds = [dstId], cur = dstId;
      while (cur !== srcId) {
        cur = prev[cur];
        if (cur === undefined) return null;
        pathIds.unshift(cur);
      }

      var totalTime = 0, totalDist = 0, interchanges = 0, prevLine = null;
      for (var i = 1; i < pathIds.length; i++) {
        var edge = viaLine[pathIds[i]];
        totalTime += edge.time;
        totalDist += edge.dist;
        if (prevLine !== null && edge.line !== prevLine) interchanges++;
        prevLine = edge.line;
      }
      return {
        pathIds: pathIds, totalTime: totalTime, totalDist: totalDist,
        interchanges: interchanges,
      };
    }

    // Mirrors src/fare.py exactly, so keep the two in sync if the slabs change.
    var FARE_SLABS = [[5, 5], [10, 10], [15, 15], [20, 15], [30, 20], [40, 25], [50, 30], [60, 35], [80, 40], [120, 50]];
    function estimateFare(km) {
      for (var i = 0; i < FARE_SLABS.length; i++) {
        if (km <= FARE_SLABS[i][0]) return FARE_SLABS[i][1];
      }
      return FARE_SLABS[FARE_SLABS.length - 1][1];
    }
    function estimateCo2Saved(km) {
      return Math.round((121.0 - 41.0) * km) / 1000;
    }

    function lineColor(line) {
      var tokens = String(line).split('/');
      for (var i = 0; i < tokens.length; i++) {
        if (LINE_COLORS[tokens[i]]) return LINE_COLORS[tokens[i]];
      }
      return LINE_COLORS.Interchange;
    }

    function renderResult(route) {
      var chips = route.pathIds.map(function (id, i) {
        var st = NET.stations[id];
        var color = lineColor(st.l);
        var arrow = i > 0 ? '<span class="stop-arrow">&rarr;</span>' : '';
        return arrow + '<span class="stop-chip' + (st.h ? ' hub' : '') + '" style="color:' + color + '">'
          + '<span class="dot" style="background:' + color + '"></span>' + st.n + '</span>';
      }).join('');

      var fareInr = estimateFare(route.totalDist);
      var co2 = estimateCo2Saved(route.totalDist);

      document.getElementById('plannerResult').innerHTML =
        '<div class="route-card">'
        + '<div class="route-card-head"><h3>' + NET.stations[route.pathIds[0]].n + ' &rarr; '
        + NET.stations[route.pathIds[route.pathIds.length - 1]].n + '</h3>'
        + '<div class="route-totals">'
        + '<span><strong>' + Math.round(route.totalTime) + '</strong> min</span>'
        + '<span><strong>' + route.totalDist.toFixed(1) + '</strong> km</span>'
        + '<span><strong>' + route.pathIds.length + '</strong> stops</span>'
        + '<span><strong>' + route.interchanges + '</strong> interchange' + (route.interchanges !== 1 ? 's' : '') + '</span>'
        + '<span><strong>&#8377;' + fareInr + '</strong> fare</span>'
        + '<span><strong>' + co2.toFixed(1) + '</strong> kg CO&#8322; saved</span>'
        + '</div></div>'
        + '<div class="ticket-tear"></div>'
        + '<div class="stop-chip-row">' + chips + '</div>'
        + '</div>';
    }

    function runPlanner() {
      var errorEl = document.getElementById('plannerError');
      errorEl.classList.remove('visible');
      document.getElementById('plannerResult').innerHTML = '';

      var fromToken = document.getElementById('plannerFrom').value;
      var toToken = document.getElementById('plannerTo').value;
      var weightKey = document.getElementById('plannerWeight').value;

      var srcId = resolveStation(fromToken);
      var dstId = resolveStation(toToken);

      if (!fromToken || !toToken) {
        errorEl.textContent = 'Enter both a "from" and a "to" station.';
        errorEl.classList.add('visible');
        return;
      }
      if (!srcId || !dstId) {
        errorEl.textContent = 'Unrecognized station name: "' + (!srcId ? fromToken : toToken) + '". Pick one from the suggestions.';
        errorEl.classList.add('visible');
        return;
      }
      if (srcId === dstId) {
        errorEl.textContent = 'Origin and destination are the same station.';
        errorEl.classList.add('visible');
        return;
      }

      var route = dijkstra(srcId, dstId, weightKey);
      if (!route) {
        errorEl.textContent = 'No route found between those stations.';
        errorEl.classList.add('visible');
        return;
      }
      renderResult(route);
    }

    var goBtn = document.getElementById('plannerGo');
    if (goBtn) {
      goBtn.addEventListener('click', runPlanner);
      ['plannerFrom', 'plannerTo'].forEach(function (id) {
        document.getElementById(id).addEventListener('keydown', function (e) {
          if (e.key === 'Enter') runPlanner();
        });
      });
      document.getElementById('plannerSwap').addEventListener('click', function () {
        var a = document.getElementById('plannerFrom'), b = document.getElementById('plannerTo');
        var tmp = a.value; a.value = b.value; b.value = tmp;
      });
    }
  })();

  // ---- chart tooltips: any element with data-tip="Title|line|line" ----
  var tip = document.createElement('div');
  tip.className = 'tip'; tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  document.addEventListener('pointermove', function (e) {
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (!el) { tip.classList.remove('on'); return; }
    var parts = el.getAttribute('data-tip').split('|');
    tip.innerHTML = '<b></b>';
    tip.firstChild.textContent = parts[0];
    parts.slice(1).forEach(function (p) { var d = document.createElement('div'); d.textContent = p; tip.appendChild(d); });
    var x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = e.clientY - tip.offsetHeight - 12;
    tip.style.left = x + 'px'; tip.style.top = (y < 8 ? e.clientY + 18 : y) + 'px';
    tip.classList.add('on');
  });

  // ---- count-up + leaderboard bar reveal on scroll ----
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('visible');
      e.target.querySelectorAll('.lb-bar').forEach(function (bar) {
        bar.style.width = bar.dataset.width + '%';
      });
      observer.unobserve(e.target);
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.fade-in').forEach(function (el) { observer.observe(el); });

  // ---- 3D force-directed network graph (real spring-layout coords) ----
  var DATA = GRAPH_DATA_JSON;
  var mount = document.getElementById('graph3d');
  if (!mount || !window.THREE) return;

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.1, 100);
  camera.position.set(0, 0, 9);

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(mount.clientWidth, mount.clientHeight);
  mount.appendChild(renderer.domElement);

  var group = new THREE.Group();
  scene.add(group);

  var starGeo = new THREE.BufferGeometry();
  var starCount = 260, starPos = new Float32Array(starCount * 3);
  for (var i = 0; i < starCount; i++) {
    starPos[i*3] = (Math.random()-0.5) * 30;
    starPos[i*3+1] = (Math.random()-0.5) * 30;
    starPos[i*3+2] = (Math.random()-0.5) * 30 - 6;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x3a372f, size: 0.04 })));

  var SCALE = 5.2;
  var idToIndex = {};
  DATA.nodes.forEach(function (n, i) { idToIndex[n.id] = i; });

  DATA.nodes.forEach(function (n) {
    var color = new THREE.Color(n.color);
    var r = 0.06 + n.size * 0.1;
    var core = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), new THREE.MeshBasicMaterial({ color: color }));
    core.position.set(n.x * SCALE, n.y * SCALE, n.z * SCALE);
    group.add(core);
    if (n.hub) {
      var halo = new THREE.Mesh(
        new THREE.SphereGeometry(r * 3, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xFFC72C, transparent: true, opacity: 0.18 })
      );
      halo.position.copy(core.position);
      group.add(halo);
    }
  });

  var edgeGeo = new THREE.BufferGeometry();
  var edgePositions = [];
  DATA.edges.forEach(function (e) {
    var a = DATA.nodes[idToIndex[e.source]], b = DATA.nodes[idToIndex[e.target]];
    edgePositions.push(a.x*SCALE, a.y*SCALE, a.z*SCALE, b.x*SCALE, b.y*SCALE, b.z*SCALE);
  });
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edgePositions), 3));
  var edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0xbdb7a6, transparent: true, opacity: 0.32 }));
  group.add(edgeLines);

  var dragging = false, lastX = 0, lastY = 0, rotY = 0.35, rotX = -0.15;
  var dom = renderer.domElement;
  dom.style.cursor = 'grab';
  dom.addEventListener('pointerdown', function (e) { dragging = true; lastX = e.clientX; lastY = e.clientY; dom.style.cursor = 'grabbing'; });
  window.addEventListener('pointerup', function () { dragging = false; dom.style.cursor = 'grab'; });
  window.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    rotY += (e.clientX - lastX) * 0.005;
    rotX += (e.clientY - lastY) * 0.005;
    lastX = e.clientX; lastY = e.clientY;
  });

  function onResize() {
    camera.aspect = mount.clientWidth / mount.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(mount.clientWidth, mount.clientHeight);
  }
  window.addEventListener('resize', onResize);

  var t = 0;
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  (function animate() {
    requestAnimationFrame(animate);
    t += 0.005;
    if (!dragging && !REDUCED) rotY += 0.0016;
    group.rotation.y = rotY;
    group.rotation.x = rotX + Math.sin(t) * 0.03;
    renderer.render(scene, camera);
  })();
})();
</script>
"""


def build_dashboard(
    stations_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    layout_3d: dict[str, tuple[float, float, float]],
    centrality_df: pd.DataFrame,
    routes: dict[str, dict],
    network_summary: dict,
    buffer_stats: dict,
    insights: dict,
) -> None:
    """Render and save the standalone dashboard to output/dashboard.html."""
    line_by_id = stations_df.set_index("station_id")["line"]
    footfall_by_id = stations_df.set_index("station_id")["daily_footfall"]

    graph_nodes = []
    for sid, (x, y, z) in layout_3d.items():
        line = str(line_by_id.get(sid, ""))
        graph_nodes.append({
            "id": sid, "x": x, "y": y, "z": z,
            "color": config.LINE_COLORS.get(line.split("/")[0], "#9aa4b8"),
            "size": min(1.0, float(footfall_by_id.get(sid, 0)) / 800_000),
            "hub": "/" in line,
        })
    graph_edges = [{"source": r.source_id, "target": r.target_id} for r in edges_df.itertuples(index=False)]
    graph_json = json.dumps({"nodes": graph_nodes, "edges": graph_edges})

    # Compact network payload for the client-side route planner: station
    # lookup by id + a flat edge list. Keys are kept short (n/l/lat/lon/h)
    # since this is inlined into the page and parsed by every visitor.
    network_stations = {
        r.station_id: {
            "n": r.name, "l": r.line, "lat": r.latitude, "lon": r.longitude,
            "h": bool(r.interchange_hub),
        }
        for r in stations_df.itertuples(index=False)
    }
    network_edges = [
        [r.source_id, r.target_id, r.line_name, r.distance_km, r.travel_time_mins]
        for r in edges_df.itertuples(index=False)
    ]
    network_json = json.dumps({"stations": network_stations, "edges": network_edges})

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = "\n".join([
        _nav_html(network_summary, generated_at),
        _marquee_html(stations_df),
        "<main>",
        _hero_html(stations_df, network_summary),
        _map_preview_html(),
        _routes_html(routes, stations_df),
        _planner_html(stations_df),
        _how_it_works_html(),
        _analytics_html(insights, buffer_stats),
        _leaderboard_html(centrality_df),
        _coverage_html(buffer_stats, network_summary),
        "</main>",
        _footer_html(generated_at),
    ])

    line_colors_json = json.dumps(config.LINE_COLORS)
    js = (
        _JS.replace("GRAPH_DATA_JSON", graph_json)
        .replace("NETWORK_DATA_JSON", network_json)
        .replace("LINE_COLORS_JSON", line_colors_json)
    )

    favicon = urllib.parse.quote(LOGO_SVG.replace(' class="logo-mark" aria-hidden="true"', ' xmlns="http://www.w3.org/2000/svg"'))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Pulse · Mumbai Suburban Rail Analytics</title>
<meta name="theme-color" content="#0B0B0C">
<link rel="icon" href="data:image/svg+xml,{favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@700&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
{body}
{js}
</body>
</html>"""

    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.DASHBOARD_HTML.write_text(html, encoding="utf-8")
        logger.info("Dashboard written to %s", config.DASHBOARD_HTML)
    except OSError as exc:
        logger.error("Failed to write dashboard HTML: %s", exc)
        raise

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
from datetime import datetime, timezone

import pandas as pd

from src import config

logger = config.get_logger(__name__)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _line_badges(line: str) -> str:
    tokens = line.split("/")
    spans = []
    for t in tokens:
        color = config.LINE_COLORS.get(t, config.LINE_COLORS["Interchange"])
        spans.append(f'<span class="badge" style="--c:{color}">{t}</span>')
    return "".join(spans)


def _stat_tile(label: str, value: str, sub: str = "", delay: int = 0) -> str:
    return f"""
    <div class="stat-tile pixel-cut" style="--d:{delay}ms">
      <div class="stat-value flicker">{value}</div>
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
        <span class="muted">/</span>
        <span title="Degree centrality">{row['degree_centrality']:.3f}</span>
      </div>
      <div class="lb-footfall">{_fmt_int(int(row['daily_footfall']))}<span class="muted">/day</span></div>
    </div>"""


def _route_svg(route: dict, stations_df: pd.DataFrame) -> str:
    """A horizontal SVG strip: station dots joined by a line, with a small
    'train' circle looping along the path via SMIL <animateMotion> -- no JS
    or extra libraries needed for this specific animation."""
    ids = route["path_ids"]
    n = len(ids)
    width = max(560, 96 * (n - 1) + 80)
    y = 34
    xs = [40 + i * (width - 80) / max(n - 1, 1) for i in range(n)]
    line_by_id = stations_df.set_index("station_id")["line"]

    dots, labels = [], []
    for i, (sid, x) in enumerate(zip(ids, xs)):
        color = config.LINE_COLORS.get(line_by_id.get(sid, "").split("/")[0], "#9aa4b8")
        is_hub = "/" in str(line_by_id.get(sid, ""))
        r = 9 if is_hub else 6
        ring = f'<circle cx="{x}" cy="{y}" r="{r+5}" fill="none" stroke="{color}" stroke-opacity="0.5"><animate attributeName="r" values="{r+4};{r+8};{r+4}" dur="1.6s" repeatCount="indefinite"/></circle>' if is_hub else ""
        dots.append(f'{ring}<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="#050212" stroke-width="2.5"/>')
        labels.append(
            f'<text x="{x}" y="{y+26}" text-anchor="middle" class="route-label">{route["path_names"][i]}</text>'
        )

    path_d = f"M{xs[0]},{y} " + " ".join(f"L{x},{y}" for x in xs[1:])
    duration = max(2.4, min(10, n * 0.9))

    return f"""
    <svg viewBox="0 0 {width} 90" class="route-svg" preserveAspectRatio="xMinYMid meet">
      <path d="{path_d}" stroke="rgba(255,255,255,0.16)" stroke-width="3" fill="none" stroke-dasharray="1,7" stroke-linecap="round"/>
      {''.join(dots)}
      {''.join(labels)}
      <circle r="6" fill="#fff">
        <animateMotion dur="{duration}s" repeatCount="indefinite" path="{path_d}"/>
      </circle>
      <circle r="12" fill="#FFE156" opacity="0.35">
        <animateMotion dur="{duration}s" repeatCount="indefinite" path="{path_d}"/>
      </circle>
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
    if beam:
        return f'<div class="beam-card"><div class="beam-inner route-card">{inner}</div></div>'
    return f'<div class="route-card pixel-cut">{inner}</div>'


_CSS = """
:root{
  --bg:#0a0714; --bg-2:#0f0a20; --surface:#140f28; --surface-2:#1b1436; --border:rgba(255,255,255,.10);
  --text:#f1eefc; --muted:#a79fc9; --faint:#5f5885;
  --western:#39FF6E; --central:#FF3B4E; --harbour:#3EC1FF; --transharbour:#FFB000; --hub:#FFE156;
  --pink:#FF2E9A; --cyan:#22F0FF; --amber:#FFB000; --violet:#B388FF;
  --primary:#FF2E9A; --primary-2:#22F0FF;
  --font-display:'VT323',monospace; --font-pixel:'Press Start 2P',cursive;
  --font-body:'Inter',sans-serif; --font-mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--bg);
  color:var(--text); font-family:var(--font-body); line-height:1.5; -webkit-font-smoothing:antialiased;
  position:relative;
}
::selection{ background:var(--pink); color:#0a0714; }
.container{max-width:1180px; margin:0 auto; padding:0 28px;}
a{color:var(--cyan);}

/* ---------- Global texture: scanlines + vignette + noise ---------- */
.scanlines{
  position:fixed; inset:0; pointer-events:none; z-index:2; mix-blend-mode:overlay; opacity:.5;
  background:repeating-linear-gradient(to bottom, rgba(255,255,255,.05) 0 1px, transparent 1px 3px);
}
.vignette{
  position:fixed; inset:0; pointer-events:none; z-index:1;
  background:radial-gradient(120% 90% at 50% -10%, rgba(255,46,154,.10), transparent 55%),
             radial-gradient(120% 90% at 100% 110%, rgba(34,240,255,.08), transparent 55%),
             radial-gradient(140% 140% at 50% 50%, transparent 55%, rgba(0,0,0,.55) 100%);
}
.meteor{
  position:fixed; top:-10%; width:2px; height:110px; z-index:1; pointer-events:none;
  background:linear-gradient(to bottom, transparent, var(--cyan) 40%, #fff 100%);
  transform:rotate(215deg); opacity:0; filter:drop-shadow(0 0 6px var(--cyan));
  animation:meteor-fall linear infinite;
}
@keyframes meteor-fall{
  0%{ transform:translate(0,-10vh) rotate(215deg); opacity:0; }
  8%{ opacity:1; }
  70%{ opacity:.6; }
  100%{ transform:translate(-60vw,90vh) rotate(215deg); opacity:0; }
}

/* ---------- Departure-board marquee ---------- */
.marquee-bar{
  position:relative; z-index:3; background:#000; border-bottom:2px solid var(--amber);
  overflow:hidden; white-space:nowrap; box-shadow:0 2px 24px rgba(255,176,0,.15);
}
.marquee-track{ display:inline-block; padding-left:100%; animation:marquee var(--marquee-dur,32s) linear infinite; }
.marquee-track span{
  font-family:var(--font-display); font-size:20px; letter-spacing:.04em; color:var(--amber);
  text-shadow:0 0 8px rgba(255,176,0,.65); padding:6px 0;
}
.marquee-track .sep{ color:var(--faint); margin:0 22px; }
@keyframes marquee{ to{ transform:translateX(-100%); } }

/* ---------- Hero ---------- */
.hero{position:relative; z-index:3; padding:56px 0 40px; overflow:hidden;}
.hero-spotlight{
  position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(500px circle at var(--mx,50%) var(--my,20%), rgba(255,46,154,.16), transparent 55%);
}
.hero-grid{display:grid; grid-template-columns:1.1fr 1fr; gap:32px; align-items:center; position:relative;}
@media (max-width:860px){ .hero-grid{grid-template-columns:1fr;} }
.eyebrow{
  font-family:var(--font-pixel); font-size:9px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--cyan); display:flex; align-items:center; gap:10px; margin-bottom:18px; line-height:1.8;
}
.eyebrow::before{content:""; width:8px; height:8px; background:var(--western); box-shadow:0 0 10px var(--western); }
h1.title{
  font-family:var(--font-display); font-weight:400; font-size:clamp(42px,6.4vw,72px);
  line-height:0.98; margin:0 0 18px; letter-spacing:.01em; color:#fff; text-shadow:3px 3px 0 rgba(255,46,154,.35);
}
.title .accent{ color:var(--western); text-shadow:3px 3px 0 rgba(57,255,110,.3); }
.lede{color:var(--muted); font-size:15.5px; max-width:52ch; margin:0 0 28px;}
.hero-canvas-wrap{position:relative; height:360px; border:2px solid var(--border); overflow:hidden; background:radial-gradient(circle at 50% 50%, #171030 0%, #0a0714 75%);}
#graph3d{position:absolute; inset:0;}
.hero-canvas-wrap canvas{display:block; width:100%; height:100%;}
.hero-canvas-tag{
  position:absolute; bottom:10px; left:10px; font-family:var(--font-mono); font-size:10px;
  color:var(--faint); background:rgba(10,7,20,.7); padding:4px 9px; border:1px solid var(--border);
}

.stat-row{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:10px;}
@media (max-width:700px){ .stat-row{grid-template-columns:repeat(2,1fr);} }
.stat-tile{
  background:var(--surface); border:2px solid var(--border); padding:16px 18px;
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
  box-shadow:4px 4px 0 rgba(0,0,0,.5); animation:pop-in .5s ease backwards; animation-delay:var(--d,0ms);
}
.stat-tile:hover{ transform:translate(-2px,-2px); border-color:var(--pink); box-shadow:6px 6px 0 rgba(255,46,154,.35); }
.stat-value{font-family:var(--font-display); font-size:38px; font-weight:400; color:#fff; line-height:1;}
.stat-label{font-size:11.5px; color:var(--muted); margin-top:6px; text-transform:uppercase; letter-spacing:.04em;}
.stat-sub{font-size:11px; color:var(--faint); margin-top:2px;}
@keyframes pop-in{ from{ opacity:0; transform:translateY(10px) scale(.97);} to{ opacity:1; transform:none;} }
@keyframes flicker{
  0%,100%{ opacity:1; } 92%{ opacity:1; } 93%{ opacity:.4; } 94%{ opacity:1; } 96%{ opacity:.7; } 97%{ opacity:1; }
}
.flicker{ animation:flicker 4.5s ease-in-out infinite; }

/* ---------- pixel-cut corners (shared) ---------- */
.pixel-cut{
  clip-path:polygon(0 10px,10px 10px,10px 0,calc(100% - 10px) 0,calc(100% - 10px) 10px,100% 10px,
    100% calc(100% - 10px),calc(100% - 10px) calc(100% - 10px),calc(100% - 10px) 100%,
    10px 100%,10px calc(100% - 10px),0 calc(100% - 10px));
}

/* ---------- Sections ---------- */
section{position:relative; z-index:3; padding:44px 0;}
.section-head{display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:22px; flex-wrap:wrap;}
h2.section-title{font-family:var(--font-display); font-size:32px; margin:0; letter-spacing:.01em; color:#fff;}
h2.section-title::before{ content:"// "; color:var(--pink); }
.section-desc{color:var(--muted); font-size:13.5px; margin:6px 0 0; max-width:60ch;}
.divider{height:2px; background:repeating-linear-gradient(90deg,var(--border) 0 8px,transparent 8px 14px);}

/* ---------- Map preview ---------- */
.map-preview{position:relative; border:2px solid var(--border); background:var(--surface); overflow:hidden;}
.map-preview iframe{display:block; width:100%; height:520px; border:0; filter:saturate(1.05);}
@media (max-width:760px){ .map-preview iframe{height:360px;} }
.map-preview-cta{
  position:absolute; top:16px; right:16px; z-index:5;
  background:rgba(10,7,20,.88); backdrop-filter:blur(10px);
  border:2px solid var(--pink); color:#fff; font-family:var(--font-display);
  font-size:14px; padding:9px 16px; text-decoration:none; display:inline-flex;
  align-items:center; gap:8px; box-shadow:4px 4px 0 rgba(255,46,154,.4);
  transition:transform .15s ease;
}
.map-preview-cta:hover{transform:translate(-2px,-2px);}
.map-preview-note{
  position:absolute; bottom:0; left:0; right:0; z-index:4; padding:10px 16px;
  background:linear-gradient(0deg,rgba(10,7,20,.9),transparent);
  font-size:11.5px; color:var(--faint); font-family:var(--font-mono);
}

/* ---------- How it works ---------- */
.how-grid{display:flex; align-items:stretch;}
@media (max-width:900px){ .how-grid{flex-direction:column; gap:14px;} }
.how-step{
  background:var(--surface); border:2px solid var(--border); padding:20px 18px; position:relative;
  flex:1 1 0; min-width:0;
}
.how-step-num{
  font-family:var(--font-display); font-size:13px; color:var(--pink); letter-spacing:.08em;
  margin-bottom:8px;
}
.how-step h4{margin:0 0 8px; font-family:var(--font-display); font-size:17px; color:#fff;}
.how-step p{margin:0; font-size:12.5px; color:var(--muted); line-height:1.55;}
.how-step code{
  display:block; margin-top:10px; font-family:var(--font-mono); font-size:11px; color:var(--cyan);
  background:var(--surface-2); border:1px solid var(--border); padding:8px 10px; overflow-x:auto; white-space:nowrap;
}
.how-arrow{
  display:flex; align-items:center; justify-content:center; color:var(--pink); font-size:20px;
  background:var(--surface); border-top:2px solid var(--border); border-bottom:2px solid var(--border);
  flex:0 0 36px;
}
@media (max-width:900px){ .how-arrow{display:none;} }

/* ---------- Route cards ---------- */
.route-grid{display:grid; grid-template-columns:1fr 1fr; gap:20px;}
@media (max-width:860px){ .route-grid{grid-template-columns:1fr;} }
.route-card{background:var(--surface); padding:22px 24px;}
.route-card-head{display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:8px; margin-bottom:14px;}
.route-card-head h3{font-family:var(--font-display); font-size:22px; margin:0; color:var(--cyan); letter-spacing:.02em;}
.route-totals{display:flex; gap:14px; font-size:12.5px; color:var(--muted);}
.route-totals strong{color:#fff; font-family:var(--font-mono); font-weight:500;}
.ticket-tear{ height:0; border-top:2px dashed var(--border); margin:14px -24px 14px; }
.route-svg-wrap{overflow-x:auto;}
.route-svg{width:100%; min-width:520px; height:80px;}
.route-label{font-size:9.5px; fill:var(--faint); font-family:var(--font-body);}

/* rotating gradient border beam, reserved for the route cards */
.beam-card{ position:relative; padding:2px; isolation:isolate; overflow:hidden; }
.beam-card::before{
  content:""; position:absolute; inset:-2px; z-index:-1;
  background:conic-gradient(from 0deg, transparent 0 55%, var(--pink), var(--cyan), var(--western), transparent 78% 100%);
  animation:beam-spin 5s linear infinite;
}
.beam-card .beam-inner{ background:var(--surface); height:100%; }
@keyframes beam-spin{ to{ transform:rotate(360deg); } }

/* ---------- Leaderboard ---------- */
.leaderboard{background:var(--surface); border:2px solid var(--border); padding:8px 22px;}
.lb-row{
  display:grid; grid-template-columns:38px 1.4fr 1fr 92px 110px; align-items:center; gap:14px;
  padding:13px 0; border-bottom:1px dashed var(--border);
}
.lb-row:last-child{border-bottom:none;}
.lb-rank{font-family:var(--font-display); color:var(--amber); font-size:22px;}
.lb-name{font-weight:600; font-size:14.5px;}
.lb-badges{margin-top:5px;}
.badge{
  --c:#999; display:inline-block; font-size:9.5px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  color:#050212; background:var(--c); padding:2px 7px; margin-right:5px;
}
.lb-bar-track{height:8px; background:var(--surface-2); overflow:hidden; border:1px solid var(--border);}
.lb-bar{height:100%; width:0; background:linear-gradient(90deg,var(--pink),var(--cyan)); transition:width 1.1s cubic-bezier(.16,1,.3,1); box-shadow:0 0 10px rgba(34,240,255,.6);}
.lb-metrics{font-family:var(--font-mono); font-size:12px; color:var(--muted); text-align:right;}
.lb-footfall{font-family:var(--font-mono); font-size:12.5px; color:#fff; text-align:right;}
.lb-footfall .muted{color:var(--faint); font-size:10px;}

/* ---------- Coverage / footer ---------- */
.coverage-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;}
@media (max-width:760px){ .coverage-grid{grid-template-columns:1fr;} }
.cta-row{display:flex; gap:14px; margin-top:26px; flex-wrap:wrap;}
.btn{
  font-family:var(--font-display); font-size:18px; padding:11px 22px;
  text-decoration:none; display:inline-flex; align-items:center; gap:8px; letter-spacing:.02em;
  transition:transform .15s ease, box-shadow .15s ease; border:2px solid transparent;
}
.btn-primary{background:var(--pink); color:#0a0714; box-shadow:4px 4px 0 rgba(255,46,154,.4);}
.btn-primary:hover{transform:translate(-2px,-2px); box-shadow:6px 6px 0 rgba(255,46,154,.5);}
.btn-ghost{background:transparent; border-color:var(--border); color:var(--text);}
.btn-ghost:hover{border-color:var(--cyan); color:var(--cyan); transform:translate(-2px,-2px);}

/* ---------- Live route planner ---------- */
.planner-card{background:var(--surface); border:2px solid var(--border); padding:22px 24px;}
.planner-row{display:grid; grid-template-columns:1fr auto 1fr auto auto; gap:12px; align-items:end;}
@media (max-width:760px){ .planner-row{grid-template-columns:1fr; } }
.planner-field label{display:block; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--faint); margin-bottom:6px;}
.planner-field input, .planner-field select{
  width:100%; background:var(--surface-2); border:2px solid var(--border); color:var(--text);
  font-family:var(--font-body); font-size:14px; padding:10px 12px; box-sizing:border-box;
}
.planner-field input:focus, .planner-field select:focus{outline:none; border-color:var(--cyan);}
.planner-swap{
  background:var(--surface-2); border:2px solid var(--border); color:var(--cyan);
  width:40px; height:40px; cursor:pointer; font-size:16px; line-height:1;
}
.planner-swap:hover{border-color:var(--cyan);}
.planner-error{color:var(--pink); font-size:13px; margin-top:12px; display:none;}
.planner-error.visible{display:block;}
.planner-result{margin-top:20px;}
.planner-result:empty{margin-top:0;}
.stop-chip-row{display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:14px;}
.stop-chip{
  display:inline-flex; align-items:center; gap:6px; font-family:var(--font-mono); font-size:11.5px;
  background:var(--surface-2); border:1px solid var(--border); padding:4px 9px 4px 6px; border-radius:999px;
}
.stop-chip .dot{width:8px; height:8px; border-radius:50%; flex:none; box-shadow:0 0 6px currentColor;}
.stop-chip.hub{border-color:#FFD24C; color:#FFD24C;}
.stop-arrow{color:var(--faint); font-size:11px;}

footer{position:relative; z-index:3; border-top:2px solid var(--border); padding:32px 0 60px; color:var(--faint); font-size:12px;}
footer h4{font-family:var(--font-display); color:var(--amber); font-size:16px; letter-spacing:.03em; margin:0 0 10px;}
footer ul{margin:0; padding-left:18px;}
footer li{margin-bottom:6px;}
footer .cols{display:grid; grid-template-columns:2fr 1fr; gap:32px;}
@media (max-width:760px){ footer .cols{grid-template-columns:1fr;} }

.fade-in{opacity:0; transform:translateY(14px); transition:opacity .6s ease, transform .6s ease;}
.fade-in.visible{opacity:1; transform:none;}

.tilt{ transition:transform .12s ease-out; transform-style:preserve-3d; will-change:transform; }
"""


def _marquee_html(stations_df: pd.DataFrame) -> str:
    items = []
    for row in stations_df.sort_values("daily_footfall", ascending=False).itertuples(index=False):
        tag = " -- INTERCHANGE" if row.interchange_hub else ""
        items.append(f'<span>{row.name.upper()}{tag}</span><span class="sep">&bull;</span>')
    strip = "".join(items)
    # A per-station-count guess for the duration was still not slow enough
    # in practice (font-width varies per name, so pixel distance doesn't
    # scale cleanly with item count). The reliable fix is to measure the
    # strip's actual rendered width at runtime and derive the duration from
    # a fixed, comfortable reading speed -- see the script at the bottom of
    # the page that sets --marquee-dur once the DOM has laid out.
    return f"""
    <div class="marquee-bar">
      <div class="marquee-track" id="marqueeTrack" style="--marquee-dur:90s">{strip}{strip}</div>
    </div>"""


def _hero_html(stations_df: pd.DataFrame, network_summary: dict) -> str:
    return f"""
    <section class="hero">
      <div class="hero-spotlight" id="heroSpotlight"></div>
      <div class="container hero-grid">
        <div>
          <div class="eyebrow">&#9679; live network analytics &middot; {len(stations_df)} stations</div>
          <h1 class="title">Mumbai Suburban Rail<br><span class="accent">Efficiency &amp; Accessibility</span></h1>
          <p class="lede">
            A graph + spatial model of the Western, Central, Harbour and Trans-Harbour lines &mdash;
            real stations, real coordinates, real topology &mdash; built to surface
            the busiest interchanges, the fastest cross-network routes, and how far
            a 1&nbsp;km walk reaches from each hub.
          </p>
        </div>
        <div class="hero-canvas-wrap">
          <div id="graph3d"></div>
          <div class="hero-canvas-tag">force-directed graph &middot; drag to look &middot; live WebGL</div>
        </div>
      </div>
      <div class="container stat-row">
        {_stat_tile("Stations modeled", str(network_summary['num_stations']), delay=0)}
        {_stat_tile("Track segments", str(network_summary['num_edges']), f"{network_summary['total_track_km']} km total", delay=80)}
        {_stat_tile("Combined daily footfall", f"{network_summary['total_daily_footfall']/1e6:.2f}M", f"riders across all {network_summary['num_stations']} stations", delay=160)}
        {_stat_tile("Interchange hubs", str(len(network_summary['interchange_hubs'])), "found via centrality, not asserted", delay=240)}
      </div>
    </section>"""


def _map_preview_html() -> str:
    """A live, embedded preview of the Folium map right on the homepage --
    the map used to be reachable only via a small text link buried in the
    Coverage section at the bottom of the page; landing on it immediately
    gives the report an actual visual centerpiece instead of stats-first."""
    return f"""
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Explore the Network</h2>
            <p class="section-desc">The full interactive map, live and embedded &mdash; pan, zoom, toggle
              lines, and click any station or hub for its stats.</p>
          </div>
        </div>
        <div class="map-preview pixel-cut">
          <a class="map-preview-cta" href="maps/mumbai_rail_map.html" target="_blank" rel="noopener">Open full map &#8599;</a>
          <iframe src="maps/mumbai_rail_map.html" loading="lazy" title="Mumbai Suburban Rail interactive map"></iframe>
          <div class="map-preview-note">Real coordinates &middot; real topology &middot; Leaflet + OpenStreetMap</div>
        </div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _how_it_works_html() -> str:
    """Show the actual computation pipeline, not just its output -- each
    step names the real function/formula behind it so the numbers on this
    page read as computed, not asserted."""
    steps = [
        ("01", "Clean the data", "pandas validates every station/edge, drops duplicates, "
         "type-checks coordinates.", "load_and_clean_data()"),
        ("02", "Measure real distance", "Haversine great-circle formula over each edge's "
         "WGS84 coordinates -- not a flat-plane approximation.", "2 &middot; r &middot; asin(&radic;a)"),
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
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">How The Numbers Work</h2>
            <p class="section-desc">Every stat on this page traces back to one of these four steps &mdash;
              real algorithms over real data, not hardcoded figures.</p>
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
        "Both objectives agree on the same physical route here."
        if same else
        "Optimizing for distance takes a different interchange than optimizing for time."
    )
    origin_name = time_route["path_names"][0]
    dest_name = time_route["path_names"][-1]
    return f"""
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Route Finder &mdash; {origin_name} to {dest_name}</h2>
            <p class="section-desc">Dijkstra's algorithm over the same graph, two different edge weights. {note}</p>
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
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Live Route Planner</h2>
            <p class="section-desc">
              Pick any two of the {len(stations_df)} modeled stations -- this runs Dijkstra's algorithm
              live in your browser against the same graph, no server round-trip.
            </p>
          </div>
        </div>
        <div class="planner-card pixel-cut">
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
            <button type="button" class="btn btn-primary pixel-cut" id="plannerGo">Find route &rarr;</button>
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
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Centrality Leaderboard</h2>
            <p class="section-desc">
              Ranked by betweenness centrality (how much shortest-path traffic is forced through a station),
              with degree centrality alongside. Dadar and Kurla surface on their own &mdash; nobody hardcoded them.
            </p>
          </div>
        </div>
        <div class="leaderboard">{rows}</div>
      </div>
    </section>
    <div class="container"><div class="divider"></div></div>"""


def _coverage_html(buffer_stats: dict, network_summary: dict) -> str:
    n_hubs = len(network_summary["interchange_hubs"])
    return f"""
    <section class="fade-in">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Pedestrian Accessibility Coverage</h2>
            <p class="section-desc">1 km walking-distance buffers around every algorithmically-identified interchange hub.</p>
          </div>
        </div>
        <div class="coverage-grid">
          {_stat_tile("Interchange hubs buffered", str(n_hubs))}
          {_stat_tile("Total catchment area", f"{buffer_stats.get('total_buffer_area_km2', 0):.1f} km²", "non-overlapping, 1 km radius each")}
          {_stat_tile("Avg. node degree", f"{network_summary.get('avg_degree', 0):.2f}", "mean connections per station")}
        </div>
        <div class="cta-row">
          <a class="btn btn-primary pixel-cut" href="maps/mumbai_rail_map.html">Open interactive map &rarr;</a>
          <a class="btn btn-ghost pixel-cut" href="../data/processed/interchange_buffers.geojson">Download buffers.geojson</a>
        </div>
      </div>
    </section>"""


def _footer_html(generated_at: str) -> str:
    return f"""
    <footer>
      <div class="container cols">
        <div>
          <h4>Data sources &amp; methodology</h4>
          <ul>
            <li>Station identity, line membership and adjacency order &mdash; Wikipedia
              "Western/Central/Harbour line (Mumbai Suburban Railway)" articles.</li>
            <li>Coordinates &mdash; each station's Wikipedia infobox (WGS84).</li>
            <li>Kalyan (800k/day, FY24 highest outward footfall in India) and Borivali
              (287k/day, busiest on Western Line) footfall figures are directly cited
              public statistics; other stations are documented analytical estimates
              &mdash; see README for full citations and caveats.</li>
            <li>distance_km is a real haversine calculation over the coordinates above;
              travel_time_mins is derived from that distance using a published average
              suburban EMU speed (38 km/h incl. halts).</li>
          </ul>
        </div>
        <div>
          <h4>Generated</h4>
          <p>{generated_at}</p>
          <h4>Stack</h4>
          <p>pandas &middot; networkx &middot; geopandas &middot; folium &middot; three.js</p>
        </div>
      </div>
    </footer>"""


_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
  // ---- departure-board marquee: derive speed from actual pixel width ----
  // A guessed duration (by station count, or a flat constant) drifts wrong
  // as soon as station names get longer/shorter or the count changes -- the
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

    // Plain O(V^2) Dijkstra -- the network has ~100-150 nodes, so a heap
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

    // Mirrors src/fare.py exactly -- keep the two in sync if the slabs change.
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
        '<div class="route-card pixel-cut">'
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

  // ---- cursor spotlight over the hero ----
  var hero = document.querySelector('.hero');
  if (hero) {
    hero.addEventListener('pointermove', function (e) {
      var r = hero.getBoundingClientRect();
      hero.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100) + '%');
      hero.style.setProperty('--my', ((e.clientY - r.top) / r.height * 100) + '%');
    });
  }

  // ---- meteors: a handful of diagonal streaks with staggered timing ----
  var field = document.body;
  var meteorSpecs = [
    { left: '10%', delay: '0s', dur: '6s' }, { left: '30%', delay: '1.6s', dur: '7.5s' },
    { left: '55%', delay: '3.1s', dur: '5.5s' }, { left: '75%', delay: '4.4s', dur: '8s' },
    { left: '90%', delay: '2.2s', dur: '6.8s' },
  ];
  meteorSpecs.forEach(function (m) {
    var el = document.createElement('div');
    el.className = 'meteor';
    el.style.left = m.left;
    el.style.animationDelay = m.delay;
    el.style.animationDuration = m.dur;
    field.appendChild(el);
  });

  // ---- 3D tilt on route cards ----
  document.querySelectorAll('.tilt').forEach(function (card) {
    card.addEventListener('pointermove', function (e) {
      var r = card.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width - 0.5;
      var py = (e.clientY - r.top) / r.height - 0.5;
      card.style.transform = 'perspective(800px) rotateX(' + (-py * 6) + 'deg) rotateY(' + (px * 6) + 'deg)';
    });
    card.addEventListener('pointerleave', function () { card.style.transform = ''; });
  });

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
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x4a3a7a, size: 0.05 })));

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
        new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.22 })
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
  var edgeLines = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x6a4fb8, transparent: true, opacity: 0.6 }));
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
  (function animate() {
    requestAnimationFrame(animate);
    t += 0.005;
    if (!dragging) rotY += 0.0016;
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
        '<div class="scanlines"></div><div class="vignette"></div>',
        _marquee_html(stations_df),
        _hero_html(stations_df, network_summary),
        _map_preview_html(),
        _routes_html(routes, stations_df),
        _planner_html(stations_df),
        _how_it_works_html(),
        _leaderboard_html(centrality_df),
        _coverage_html(buffer_stats, network_summary),
        _footer_html(generated_at),
    ])

    line_colors_json = json.dumps(config.LINE_COLORS)
    js = (
        _JS.replace("GRAPH_DATA_JSON", graph_json)
        .replace("NETWORK_DATA_JSON", network_json)
        .replace("LINE_COLORS_JSON", line_colors_json)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mumbai Suburban Rail &mdash; Efficiency &amp; Accessibility Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=VT323&family=Press+Start+2P&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
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

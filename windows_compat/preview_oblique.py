"""Offline interactive comparison of a source OFF and reconstructed PLY(s).

Produces one self-contained HTML file (Plotly.js inlined, no network) with a
shared 3-D coordinate frame.  Layers are toggled with checkboxes on the right,
the metrics panel on the left is collapsible, every mesh has an optional
wireframe overlay, and two range sliders control the opacity of the
reconstructed layers and of the source OFF independently.

Usage:
  python windows_compat/preview_oblique.py \
      --off data/oblique/anhui/smid_004.off \
      --ply "PSDR refined=.../sm4_refined/smid_004_psdr_surface.ply" \
      --ply "PSDR detect=.../sm4_norefine/smid_004_psdr_surface.ply" \
      --out .../compare_smid_004.html --metrics-file ...json

The legacy --new / --old form still works.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.offline import plot  # noqa: F401  (ensures the offline bundle is present)

PALETTE = ["#e2603f", "#3d7fc1", "#2e9e6b", "#a05cc7", "#d99b20"]
SOURCE_COLOR = "#8b98a8"
WIRE_SOURCE = "rgba(90,105,125,0.38)"
WIRE_LAYER = "rgba(24,34,48,0.55)"
DIV_ID = "compod-plot"


def read_off(path):
    """Return (vertices, triangulated_faces) for an ASCII OFF."""
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        first = fh.readline().strip()
        if first.lower().startswith("off") and len(first) > 3:
            counts = first[3:].split()
        else:
            counts = fh.readline().split()
        nv, nf = int(counts[0]), int(counts[1])
        verts = np.empty((nv, 3), dtype=np.float64)
        for i in range(nv):
            parts = fh.readline().split()[:3]
            verts[i] = [float(x) for x in parts]
        faces = []
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            n = int(parts[0])
            if n < 3:
                continue
            idx = [int(x) for x in parts[1:1 + n]]
            for k in range(1, n - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return verts, np.asarray(faces, dtype=np.int64)


def read_ply(path):
    m = trimesh.load(path, process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] < 3:
        raise ValueError("unexpected PLY faces: %r" % (f.shape,))
    if f.shape[1] == 3:
        tris = f
    else:
        tris = np.concatenate(
            [np.stack([f[:, 0], f[:, k], f[:, k + 1]], axis=1)
             for k in range(1, f.shape[1] - 1)], axis=0)
    return v, tris


def decimate(v, f, max_faces, rng):
    """Area-weighted triangle sampling + vertex compaction."""
    if len(f) <= max_faces:
        return v, f
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                   tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    p = np.maximum(area, 1e-12)
    p /= p.sum()
    pick = rng.choice(len(f), size=max_faces, replace=False, p=p)
    sub = f[pick]
    used, inv = np.unique(sub.ravel(), return_inverse=True)
    return v[used], inv.reshape(-1, 3)


def mesh_trace(name, v, f, color, opacity, visible=True):
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name, visible=visible,
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.9),
        hoverinfo="skip", showlegend=False,
    )


def wire_trace(name, v, f, color, max_faces, rng, visible=False):
    """Unique triangle edges as one Scatter3d 'lines' trace (NaN separated)."""
    wv, wf = decimate(v, f, max_faces, rng)
    edges = np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    n = len(edges)
    xyz = np.full((n * 3, 3), np.nan, dtype=np.float64)
    seg = np.arange(n) * 3
    xyz[seg, :] = wv[edges[:, 0], :]
    xyz[seg + 1, :] = wv[edges[:, 1], :]
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="lines", name=name + " (wire)", visible=visible,
        line=dict(color=color, width=1),
        hoverinfo="skip", showlegend=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--new", default="", help="reconstructed PLY (legacy form)")
    ap.add_argument("--old", default="", help="optional baseline PLY (legacy form)")
    ap.add_argument("--ply", action="append", default=[],
                    help="label=path, repeatable; overrides --new/--old")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=18000)
    ap.add_argument("--wire-faces", type=int, default=4000,
                    help="max triangles per wireframe overlay (keeps HTML small)")
    ap.add_argument("--opacity", type=int, default=100,
                    help="initial opacity of reconstructed layers, percent (5-100)")
    ap.add_argument("--source-opacity", type=int, default=25,
                    help="initial opacity of the source OFF shell, percent (5-100)")
    ap.add_argument("--metrics-file", default="", help="JSON with optional notes/metrics")
    ap.add_argument("--title", default="COMPOD reconstruction comparison")
    a = ap.parse_args()

    layers = []
    if a.ply:
        for item in a.ply:
            lab, sep, path = item.partition("=")
            if not sep or not path.strip():
                ap.error("bad --ply %r, expected label=path" % item)
            layers.append((lab.strip(), path.strip()))
    else:
        if not a.new:
            ap.error("need --new, or --ply label=path")
        layers.append(("new", a.new))
        if a.old:
            layers.append(("old", a.old))

    op_main = max(5, min(100, a.opacity))
    op_src = max(5, min(100, a.source_opacity))
    rng = np.random.default_rng(0)
    vo, fo = read_off(a.off)
    center = (vo.min(0) + vo.max(0)) / 2.0
    scale = float(np.linalg.norm(vo.max(0) - vo.min(0)))
    vo_n, fo_n = decimate(vo, fo, a.max_faces, rng)
    vo_n = (vo_n - center) / scale

    # trace order: for each layer (source first) -> [mesh, wire]
    traces = [
        mesh_trace("source OFF", vo_n, fo_n, SOURCE_COLOR, op_src / 100.0,
                   visible=True),
        wire_trace("source OFF", vo_n, fo_n, WIRE_SOURCE, a.wire_faces, rng,
                   visible=True),
    ]
    stats = []
    items = [("source OFF", SOURCE_COLOR, True,
              os.path.basename(a.off), len(vo), len(fo), len(fo_n))]
    for i, (lab, path) in enumerate(layers):
        v, f = read_ply(path)
        vn, fn = decimate(v, f, a.max_faces, rng)
        vn = (vn - center) / scale
        color = PALETTE[i % len(PALETTE)]
        show = (i == 0)
        traces.append(mesh_trace(lab, vn, fn, color, op_main / 100.0,
                                 visible=show))
        traces.append(wire_trace(lab, vn, fn, WIRE_LAYER, a.wire_faces, rng,
                                 visible=show))
        stats.append((lab, os.path.basename(path), len(v), len(f)))
        items.append((lab, color, show, os.path.basename(path), len(v), len(f),
                      len(fn)))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=a.title, x=0.02, font=dict(size=16, color="#24292f")),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
        height=820,
    )

    metrics = {}
    if a.metrics_file and os.path.isfile(a.metrics_file):
        with open(a.metrics_file, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)

    bbox_txt = ("bbox %.1f x %.1f x %.1f (source units); centre [%.1f, %.1f, %.1f]"
                % (*(vo.max(0) - vo.min(0)), *center))
    rows = ["<b>legend</b> &nbsp; "
            "<span style='color:%s'>&#9632;</span> source OFF" % SOURCE_COLOR]
    for i, (lab, _) in enumerate(layers):
        rows[0] += (" &nbsp; <span style='color:%s'>&#9632;</span> %s"
                    % (PALETTE[i % len(PALETTE)], lab))
    rows.append("source: %s &nbsp;|&nbsp; %d V / %d F (drawn %d)" %
                (os.path.basename(a.off), len(vo), len(fo), len(fo_n)))
    for lab, base, nv, nf in stats:
        rows.append("%s: %s &nbsp;|&nbsp; %d V / %d F" % (lab, base, nv, nf))
    rows.append(bbox_txt)
    rows.append("All meshes share one coordinate frame (centred, scaled by "
                "source bbox diagonal) &mdash; overlap means alignment.")
    for k, val in metrics.items():
        rows.append("<b>%s</b>: %s" % (k, val))

    # Right: checkbox layer picker + opacity sliders.  Left: collapsible info
    # panel (closed by default so it does not cover the scene).
    # Two independent sliders: reconstructed meshes (the subject) and the
    # translucent source OFF shell used as a reference.
    slider_html = (
        "<div class='sep'></div>"
        f"<div class='slider'><div class='slabel'><span>重建层不透明度</span>"
        f"<b id='op-main-val'>{op_main}%</b></div>"
        f"<input type='range' id='op-main' min='10' max='100' step='5' "
        f"value='{op_main}'></div>"
        f"<div class='slider'><div class='slabel'><span>源 OFF 不透明度</span>"
        f"<b id='op-src-val'>{op_src}%</b></div>"
        f"<input type='range' id='op-src' min='5' max='100' step='5' "
        f"value='{op_src}'></div>"
        "<div class='hint'>拉到 100% 看得最清楚；调低可看叠加/内部。</div>")
    cb = []
    for i, (lab, color, show, *_rest) in enumerate(items):
        cb.append(
            "<label class='row'><input type='checkbox' data-layer='%d' %s>"
            "<span class='sw' style='background:%s'></span>"
            "<span>%s</span></label>"
            % (i, "checked" if show else "", color, lab))
    checkbox_html = (
        "<div id='layer-panel'>"
        "<div class='hd'>图层</div>"
        + "".join(cb)
        + slider_html +
        "<div class='sep'></div>"
        "<label class='row'><input type='checkbox' id='wire-toggle' checked>"
        "<span class='sw' style='background:transparent;border:1px solid #98a2b3'></span>"
        "<span>显示线框</span></label>"
        "<div class='quick'><button type='button' id='btn-all'>全部显示</button>"
        "<button type='button' id='btn-src'>仅源 OFF</button></div>"
        "</div>")
    info_html = ("<details id='info-panel'><summary>信息与指标（点击展开）</summary>"
                 "<div class='body'>%s</div></details>" % "<br>".join(rows))

    css = (
        "<style>"
        "#layer-panel{position:fixed;right:16px;top:56px;z-index:20;"
        "min-width:214px;background:rgba(255,255,255,.95);border:1px solid #d0d5dd;"
        "border-radius:8px;padding:9px 11px;font:12px/1.6 -apple-system,Segoe UI,sans-serif;"
        "color:#24292f;box-shadow:0 2px 8px rgba(16,24,40,.10)}"
        "#layer-panel .hd{font-weight:600;margin-bottom:5px}"
        "#layer-panel .row{display:flex;align-items:center;gap:6px;cursor:pointer;"
        "padding:2px 0;white-space:nowrap}"
        "#layer-panel .row input{margin:0;cursor:pointer}"
        "#layer-panel .sw{display:inline-block;width:10px;height:10px;border-radius:2px;"
        "flex:0 0 auto}"
        "#layer-panel .sep{height:1px;background:#e4e7ec;margin:6px 0}"
        "#layer-panel .slider{margin:3px 0 8px}"
        "#layer-panel .slabel{display:flex;justify-content:space-between;gap:10px;"
        "color:#475467;margin-bottom:3px}"
        "#layer-panel .slabel b{font-weight:600;color:#24292f;"
        "font-variant-numeric:tabular-nums}"
        "#layer-panel input[type=range]{display:block;width:100%;margin:0;"
        "accent-color:#3d7fc1;cursor:pointer}"
        "#layer-panel .hint{color:#98a2b3;font-size:11px;line-height:1.45;"
        "margin:-3px 0 2px}"
        "#layer-panel .quick{display:flex;gap:6px;margin-top:7px}"
        "#layer-panel button{flex:1;font:11px -apple-system,Segoe UI,sans-serif;"
        "padding:3px 5px;border:1px solid #d0d5dd;border-radius:5px;background:#f9fafb;"
        "color:#344054;cursor:pointer}"
        "#layer-panel button:hover{background:#f2f4f7}"
        "#info-panel{position:fixed;left:14px;top:56px;z-index:20;max-width:430px;"
        "background:rgba(255,255,255,.95);border:1px solid #d0d5dd;border-radius:8px;"
        "font:12px/1.65 -apple-system,Segoe UI,sans-serif;color:#24292f;"
        "box-shadow:0 2px 8px rgba(16,24,40,.10)}"
        "#info-panel summary{cursor:pointer;padding:7px 11px;font-weight:600;list-style:none;"
        "white-space:nowrap}"
        "#info-panel summary::-webkit-details-marker{display:none}"
        "#info-panel summary::before{content:'▸ ';color:#667085}"
        "#info-panel[open] summary::before{content:'▾ '}"
        "#info-panel .body{padding:0 12px 11px;max-height:70vh;overflow:auto;"
        "border-top:1px solid #e4e7ec}"
        "</style>")

    js = (
        "<script>"
        "function plotEl(){return document.getElementById('__DIV_ID__');}"
        "function applyVisibility(){"
        "  var gd=plotEl();"
        "  if(!gd||!gd.data)return;"
        "  var wire=document.getElementById('wire-toggle').checked;"
        "  var boxes=document.querySelectorAll('#layer-panel input[data-layer]');"
        "  var vis=[];"
        "  boxes.forEach(function(b){var on=b.checked;vis.push(on);vis.push(on&&wire);});"
        "  Plotly.restyle(gd,{visible:vis});"
        "}"
        "function applyOpacity(){"
        "  var gd=plotEl();"
        "  if(!gd||!gd.data)return;"
        "  var m=+document.getElementById('op-main').value;"
        "  var s=+document.getElementById('op-src').value;"
        "  document.getElementById('op-main-val').textContent=m+'%';"
        "  document.getElementById('op-src-val').textContent=s+'%';"
        "  var idx=[],val=[];"
        "  gd.data.forEach(function(t,i){"
        "    if(t.type==='mesh3d'){"
        "      idx.push(i);val.push((t.name==='source OFF'?s:m)/100);"
        "    }});"
        "  if(idx.length)Plotly.restyle(gd,{opacity:val},idx);"
        "}"
        "function bindPanels(){"
        "  document.querySelectorAll('#layer-panel input[data-layer],#wire-toggle')"
        "    .forEach(function(b){b.addEventListener('change',applyVisibility);});"
        "  ['op-main','op-src'].forEach(function(id){"
        "    document.getElementById(id).addEventListener('input',applyOpacity);});"
        "  document.getElementById('btn-all').addEventListener('click',function(){"
        "    document.querySelectorAll('#layer-panel input[data-layer]').forEach("
        "      function(b){b.checked=true;});applyVisibility();});"
        "  document.getElementById('btn-src').addEventListener('click',function(){"
        "    document.querySelectorAll('#layer-panel input[data-layer]').forEach("
        "      function(b,i){b.checked=(i===0);});applyVisibility();});"
        "  applyVisibility();"
        "  applyOpacity();"
        "}"
        "if(document.readyState==='loading'){"
        "  window.addEventListener('DOMContentLoaded',bindPanels);"
        "}else{bindPanels();}"
        "</script>").replace("__DIV_ID__", DIV_ID)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.write_html(a.out, include_plotlyjs="inline", full_html=True,
                   div_id=DIV_ID,
                   config={"displaylogo": False, "scrollZoom": True})
    with open(a.out, "r", encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("</body>", css + info_html + checkbox_html + js + "\n</body>", 1)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", a.out, "%.1f MB" % (os.path.getsize(a.out) / 1e6),
          "traces", len(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 Beestream Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates docs/images/architecture-light.svg, jitcast's architecture diagram.

Usage: python3 docs/images/architecture.py docs/images
"""
import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
W, H = 1400, 744
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"
T = dict(bg="#FFFFFF", ink="#0B1F33", sub="#4A5B6C", panel="#F7FAFD", box="#FFFFFF", line="#D6E1EC",
         strong="#EAF2FB", strong_ink="#0B1F33", arrow="#9AAABB")
ORANGE, AMBER, TEAL, VIOLET, BLUE = "#FF7A1A", "#FFB36B", "#4FD1C5", "#8B7CF6", "#4C9AFF"

FAMILIES = [
    ("TiRex-2", "NXAI · xLSTM · 82.5M", ORANGE,
     ["trend differencing, band calibration,", "batch assembly and padding"], "mLSTM · sLSTM · variate attention"),
    ("Timestral", "Mistral AI · transformer · 5M–209M", AMBER,
     ["blocks, scaler, sign-flip TTA,", "categorical target encoding"], "RoPE · time & variate attention · gate"),
    ("Chronos-2", "Amazon · T5 encoder · 28M, 120M", TEAL,
     ["instance norm, long-horizon unrolling,", "quantile interpolation"], "RoPE · time & group attention"),
    ("TimesFM 2.5", "Google · decoder-only · 200M", VIOLET,
     ["running-stat normalization, fed-back", "decoding, flip averaging"], "RoPE + qk-norm · causal attention"),
    ("t0", "The Forecasting Company · 102M, 256M", BLUE,
     ["causal asinh scaler, rollout,", "quantile interpolation, adjustments"], "xPos RoPE · time & group attention"),
]


def svg():
    o = []
    a = o.append

    def rect(x, y, w, h, fill, stroke=None, rx=10, sw=1.2):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{s}/>')

    def text(x, y, s, size=13, fill=None, weight=400, anchor="middle", font=FONT, italic=False):
        st = ' font-style="italic"' if italic else ""
        a(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
          f'fill="{fill or T["ink"]}" text-anchor="{anchor}"{st}>{escape(s)}</text>')

    def box(x, y, w, h, title, lines=(), accent=None, mono_title=False, size=13):
        rect(x, y, w, h, T["box"], T["line"])
        if accent:
            a(f'<rect x="{x}" y="{y + 8}" width="4" height="{h - 16}" rx="2" fill="{accent}"/>')
        cy = y + (h - (16 + 15 * len(lines))) / 2 + 13
        text(x + w / 2 + (2 if accent else 0), cy, title, size, weight=600, font=MONO if mono_title else FONT)
        for i, ln in enumerate(lines):
            text(x + w / 2, cy + 17 + 15 * i, ln, 11.5, T["sub"])

    def arrow(x1, y1, x2, y2):
        a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{T["arrow"]}" stroke-width="1.6" marker-end="url(#ah)"/>')

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
      f'aria-label="jitcast architecture">')
    a('<title>jitcast architecture</title>')
    a(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
      f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{T["arrow"]}"/></marker></defs>')
    rect(0, 0, W, H, T["bg"], rx=16)
    m = 24
    iw = W - 2 * m

    y = 20
    text(m, y + 12, "CALLERS", 11, T["sub"], 700, "start")
    cw = (iw - 2 * 16) / 3
    callers = [("Java application", ["any JVM program, one Java API"]),
               ("Python harnesses", ["fev-bench, GIFT-Eval via TiRexServer /", "TimestralServer (local socket)"]),
               ("Benchmarks", ["TiRexBench, TimestralBench,", "Chronos2Bench, cold start"])]
    for i, (ti, ls) in enumerate(callers):
        box(m + i * (cw + 16), y + 20, cw, 58, ti, ls)
        arrow(m + i * (cw + 16) + cw / 2, y + 80, m + i * (cw + 16) + cw / 2, y + 100)

    y = 122
    rect(m, y, iw, 44, T["strong"], T["line"], rx=10)
    text(W / 2, y + 20, "api · ForecastEngine.forecast(List<TimeSeries>, horizon) → List<Forecast>", 12.5,
         T["strong_ink"], 700, font=MONO)
    text(W / 2, y + 36, "targets, past and known-future covariates (NaN = missing)  →  quantiles per series and step",
         11.5, T["strong_ink"])

    y = 186
    text(m, y + 12, "MODEL FAMILIES", 11, T["sub"], 700, "start")
    fy, fh = y + 22, 318
    fw = (iw - (len(FAMILIES) - 1) * 16) / len(FAMILIES)
    for i, (name, sub, col, host, kern) in enumerate(FAMILIES):
        x = m + i * (fw + 16)
        rect(x, fy, fw, fh, T["panel"], T["line"], rx=12)
        a(f'<rect x="{x}" y="{fy}" width="{fw}" height="6" rx="3" fill="{col}"/>')
        text(x + 16, fy + 28, name, 16, weight=800, anchor="start")
        text(x + 16, fy + 45, sub, 11, T["sub"], anchor="start")
        bx, bw = x + 12, fw - 24
        box(bx, fy + 56, bw, 40, "Config + Weights", ["safetensors, every tensor bound by name"], col)
        box(bx, fy + 104, bw, 62, "Host pipeline (Java)", host, col)
        arrow(x + fw / 2, fy + 168, x + fw / 2, fy + 184)
        text(x + fw / 2, fy + 198, "forward-pass interface", 11, T["sub"], 600, italic=True)
        hw = (bw - 10) / 2
        box(bx, fy + 208, hw, 50, "CPU reference", ["FP32 oracle"], col, size=12)
        box(bx + hw + 10, fy + 208, hw, 50, "TornadoVM engine", ["plan per shape"], col, size=12)
        box(bx, fy + 266, bw, 40, "Family kernels", [kern], col)

    y = fy + fh + 18
    text(m, y + 12, "SHARED", 11, T["sub"], 700, "start")
    sy = y + 22
    widths = [200, 230, 200, iw - 200 - 230 - 200 - 3 * 12]
    shared = [("io", ["SafeTensors (mmap)"]), ("host", ["SeriesWindows · SignFlip"]), ("cpu", ["Ops: FP32 operators"]),
              ("tornado", ["Kernels: GEMMs (tiled, blocked, simdgroup, FP16, CUDA Tile)",
                           "RMSNorm · SwiGLU · GemmMode · PlanCache"])]
    x = m
    for (ti, ls), w in zip(shared, widths):
        box(x, sy, w, 58, ti, ls, mono_title=True)
        x += w + 12

    y = sy + 58 + 18
    text(m, y + 12, "RUNS ON", 11, T["sub"], 700, "start")
    ry = y + 22
    jw = 200
    box(m, ry, jw, 62, "JVM", ["CPU engines, Java 22+"])
    tx = m + jw + 12
    tw = iw - jw - 12
    rect(tx, ry, tw, 62, T["panel"], T["line"], rx=10)
    text(tx + 14, ry + 20, "TornadoVM 8", 13, weight=700, anchor="start")
    text(tx + 14, ry + 36, "KernelContext kernels,", 11, T["sub"], anchor="start")
    text(tx + 14, ry + 50, "explicit worker grids", 11, T["sub"], anchor="start")
    bk = [("CUDA", "+ cuBLAS · CUDA Tile · graphs"), ("OpenCL", "any OpenCL GPU"), ("Metal", "simdgroup matrices")]
    bx0 = tx + 180
    bw = (tw - 180 - 12 - 2 * 10) / 3
    for i, (ti, ls) in enumerate(bk):
        box(bx0 + i * (bw + 10), ry + 8, bw, 46, ti, [ls])
    a('</svg>')
    return "\n".join(o)


with open(f"{OUT}/architecture-light.svg", "w") as f:
    f.write(svg() + "\n")

#!/usr/bin/env python3
"""Generate self-contained interactive HTML demos of the article analyses.

Each demo page embeds everything it needs (audio as a base64 data URI and
the precomputed analysis results as JSON), so the files can be opened
directly in any browser -- no server, no internet access, no JavaScript
dependencies. The interactive features (playback cursor, hover readouts,
track toggles) are plain canvas drawing.

Usage::

    python scripts/make_demos.py [--output-dir demos]

Pages written (default ``demos/``):

- ``index.html``             demo gallery
- ``demo_formants.html``     Praat formant tracks (N=4 / N=6 / merged) + VTL
- ``demo_spectrogram.html``  spectrogram with formant overlays + playback
- ``demo_f0.html``           F0 contours (filtered vs. raw AC) per wahoo
- ``demo_glottal.html``      glottal pulse, train and phase portrait
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import (
    estimate_f0_praat,
    extract_glottal_pulse,
    extract_glottal_train,
    extract_praat_formants,
    iaif_sliding_analysis,
    load_audio,
    normalize_amplitude,
    reby_mccomb_vtl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DB_RANGE = 70.0  # dB displayed below the spectrogram maximum


def wav_data_uri(path):
    """Encode a WAV file as a base64 data URI for embedding in HTML."""
    raw = Path(path).read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def quantize_db(matrix, band_mask, db_range=SPEC_DB_RANGE):
    """Quantize a dB matrix to uint8 over the selected band.

    The display floor is set adaptively to ``max - db_range`` (matplotlib
    autoscaling behaviour), so recordings with a high noise floor keep
    their contrast. Values at or below the floor map to 0, the maximum
    maps to 255.
    """
    band = matrix[band_mask]
    floor = float(np.nanmax(band)) - db_range
    clipped = np.clip(band, floor, None)
    return np.round((clipped - floor) / db_range * 255.0).astype(int)


def r2(values):
    """Round an array to 2 decimals and return a nested list (JSON size)."""
    return np.round(np.asarray(values, dtype=float), 2).tolist()


def r3(values):
    """Round an array to 3 decimals and return a list (JSON size)."""
    return np.round(np.asarray(values, dtype=float), 3).tolist()


# ==========================================================================
# Shared CSS / JS blocks injected in every page
# ==========================================================================

COMMON_CSS = """
:root { --ink:#1a1a1a; --muted:#666; --line:#ddd; --accent:#0a6ebd; }
* { box-sizing: border-box; }
body { font-family: Arial, Helvetica, sans-serif; color: var(--ink);
       max-width: 980px; margin: 0 auto; padding: 16px; background:#fafafa; }
h1 { font-size: 20px; margin: 8px 0 2px; }
h2 { font-size: 15px; margin: 14px 0 6px; }
p.doc { color: var(--muted); font-size: 13px; line-height: 1.45; margin: 6px 0 12px; }
.card { background:#fff; border:1px solid var(--line); border-radius:8px;
        padding:12px 14px; margin:12px 0; }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0; }
.controls button, .controls select {
  font-size:13px; padding:5px 10px; border:1px solid var(--line);
  border-radius:6px; background:#fff; cursor:pointer; }
.controls button.on { background: var(--accent); color:#fff; border-color: var(--accent); }
.badge { font-size:13px; background:#eef4fa; border:1px solid #cfe0f0;
         border-radius:6px; padding:4px 8px; }
canvas { display:block; width:100%; height:auto; background:#fff;
         border:1px solid var(--line); border-radius:4px; touch-action:none; }
#readout { font-size:13px; color:var(--muted); min-height:18px; margin-top:4px;
           font-family:Consolas,monospace; }
a.back { font-size:13px; color: var(--accent); text-decoration:none; }
.playbtn { min-width:84px; }
"""

COMMON_JS = """
// ---------- tiny plotting helpers (vanilla canvas) --------------------
function makeCanvas(id, cssW, cssH) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  cv.style.height = cssH + "px";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { cv, ctx, W: cssW, H: cssH };
}

// Plot area inside the canvas (leaves room for labels).
const M = { l: 52, r: 14, t: 12, b: 34 };

function xScale(t0, t1) { return t => M.l + (t - t0) / (t1 - t0) * (this.W - M.l - M.r); }
function yScale(v0, v1) { return v => this.H - M.b - (v - v0) / (v1 - v0) * (this.H - M.t - M.b); }

function drawFrame(g, t0, t1, v0, v1, xlab, ylab, xticks, yticks) {
  const ctx = g.ctx, X = xScale.call(g, t0, t1), Y = yScale.call(g, v0, v1);
  ctx.clearRect(0, 0, g.W, g.H);
  ctx.font = "11px Arial"; ctx.fillStyle = "#666";
  ctx.strokeStyle = "#ddd"; ctx.lineWidth = 1;
  xticks.forEach(t => {
    ctx.beginPath(); ctx.moveTo(X(t), g.H - M.b); ctx.lineTo(X(t), M.t); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(fmt(t), X(t), g.H - M.b + 14);
  });
  yticks.forEach(v => {
    ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(g.W - M.r, Y(v)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(v), M.l - 6, Y(v) + 4);
  });
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
  ctx.strokeRect(M.l, M.t, g.W - M.l - M.r, g.H - M.t - M.b);
  ctx.fillStyle = "#333"; ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.fillText(xlab, (M.l + g.W - M.r) / 2, g.H - 6);
  ctx.save(); ctx.translate(14, (M.t + g.H - M.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylab, 0, 0); ctx.restore();
}

function fmt(v) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

function plotLine(g, xs, ys, t0, t1, v0, v1, color, width, dash) {
  const X = xScale.call(g, t0, t1), Y = yScale.call(g, v0, v1);
  const ctx = g.ctx;
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, g.W - M.l - M.r, g.H - M.t - M.b); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || isNaN(ys[i])) { pen = false; continue; }
    const px = X(xs[i]), py = Y(ys[i]);
    if (!pen) { ctx.moveTo(px, py); pen = true; } else { ctx.lineTo(px, py); }
  }
  ctx.stroke(); ctx.restore();
}

function plotDots(g, xs, ys, t0, t1, v0, v1, color, r) {
  const X = xScale.call(g, t0, t1), Y = yScale.call(g, v0, v1);
  const ctx = g.ctx;
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, g.W - M.l - M.r, g.H - M.t - M.b); ctx.clip();
  ctx.fillStyle = color;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || isNaN(ys[i])) continue;
    ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), r, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.restore();
}

// ---------- audio playback + synchronized cursor -----------------------
function setupAudio(audioEl, g, t0, t1, drawFun, cursorColor) {
  const playBtn = document.getElementById("play");
  let playing = false;
  playBtn.onclick = () => {
    if (audioEl.paused) { audioEl.play(); playBtn.textContent = "\\u23F8 Pause"; }
    else { audioEl.pause(); playBtn.textContent = "\\u25B6 Play"; }
  };
  audioEl.onended = () => { playBtn.textContent = "\\u25B6 Play"; };

  function loop() {
    const t = audioEl.currentTime;
    if (t >= t0 && t <= t1) {
      drawFun();  // full redraw (cheap at these sizes)
      const X = xScale.call(g, t0, t1);
      const ctx = g.ctx;
      ctx.save();
      ctx.strokeStyle = cursorColor || "#e63946"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(t), M.t); ctx.lineTo(X(t), g.H - M.b); ctx.stroke();
      ctx.restore();
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // Click on the plot seeks the audio.
  g.cv.addEventListener("click", ev => {
    const rect = g.cv.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const frac = (px - M.l) / (g.W - M.l - M.r);
    if (frac >= 0 && frac <= 1) {
      audioEl.currentTime = t0 + frac * (t1 - t0);
      drawFun();
    }
  });
}

// Hover readout: nearest time index -> text via cb(t, i)
function setupHover(g, t0, t1, outEl, cb) {
  g.cv.addEventListener("mousemove", ev => {
    const rect = g.cv.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const frac = (px - M.l) / (g.W - M.l - M.r);
    if (frac < 0 || frac > 1) { outEl.textContent = ""; return; }
    const t = t0 + frac * (t1 - t0);
    outEl.textContent = cb(t);
  });
  g.cv.addEventListener("mouseleave", () => { outEl.textContent = ""; });
}

function nearest(times, t) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < times.length; i++) {
    const d = Math.abs(times[i] - t);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
"""

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%%TITLE%% - baboon-acoustics demo</title>
<style>%%CSS%%</style>
</head>
<body>
<a class="back" href="index.html">&larr; All demos</a>
<h1>%%TITLE%%</h1>
<p class="doc">%%DOC%%</p>
"""

HTML_FOOT = """
<p class="doc">Generated by <code>scripts/make_demos.py</code> &mdash; audio and
analysis data are embedded in this page; it works offline.</p>
<script>%%JS%%</script>
</body>
</html>
"""


def page(title, doc, body, js, css_extra=""):
    """Assemble one demo page from blocks."""
    return (HTML_HEAD.replace("%%TITLE%%", title).replace("%%DOC%%", doc)
            .replace("%%CSS%%", COMMON_CSS + css_extra)
            + body
            + HTML_FOOT.replace("%%JS%%", COMMON_JS + js))


AUDIO_CONTROLS = """
<div class="controls">
  <button id="play" class="playbtn">&#9654; Play</button>
  <span class="badge" id="vtlbadge">%%BADGE%%</span>
  <span id="readout"></span>
</div>
<audio id="audio" src="%%AUDIO%%" preload="auto"></audio>
"""


# ==========================================================================
# Demo 1 -- formant tracks (Figure 1)
# ==========================================================================

FORMANTS_BODY = AUDIO_CONTROLS + """
<div class="card">
  <div class="controls">
    <button id="b_n4" class="on">N = 4</button>
    <button id="b_n6" class="on">N = 6</button>
    <button id="b_merged" class="on">Merged</button>
    <button id="b_ref" class="on">Reference</button>
  </div>
  <canvas id="plot"></canvas>
  <p class="doc">Solid colored lines: Parselmouth estimates (click the
  buttons to toggle). Grey dashed lines: reference data digitized from
  Bo&euml; et al. (2026). Hover to read formant values; click to seek the
  audio; the red cursor follows playback.</p>
</div>
"""

FORMANTS_JS = """
const D = __DATA__;
const g = makeCanvas("plot", 900, 420);
const audio = document.getElementById("audio");
const T0 = 0, T1 = 0.62, V0 = 0, V1 = 6.5;
const show = { n4: true, n6: true, merged: true, ref: true };

function draw() {
  drawFrame(g, T0, T1, V0, V1, "Time (s)", "Frequency (kHz)",
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 2, 3, 4, 5, 6]);
  if (show.ref)
    for (let k = 0; k < 4; k++)
      plotLine(g, D.ref_t, D.ref[k], T0, T1, V0, V1, "#999", 1.2, [5, 4]);
  if (show.n4)
    for (let k = 0; k < 4; k++)
      plotLine(g, D.t, D.n4_khz[k], T0, T1, V0, V1, "#0a6ebd", 1.6);
  if (show.n6)
    for (let k = 0; k < 4; k++)
      plotLine(g, D.t, D.n6_khz[k], T0, T1, V0, V1, "#2a9d8f", 1.6);
  if (show.merged)
    for (let k = 0; k < 4; k++) {
      plotLine(g, D.t_merged, D.merged_khz[k], T0, T1, V0, V1, "#e76f51", 1.8);
      plotDots(g, D.t_merged, D.merged_khz[k], T0, T1, V0, V1, "#e76f51", 3);
    }
}
function bind(id, key) {
  document.getElementById(id).onclick = ev => {
    show[key] = !show[key];
    ev.target.classList.toggle("on", show[key]);
    draw();
  };
}
bind("b_n4", "n4"); bind("b_n6", "n6"); bind("b_merged", "merged"); bind("b_ref", "ref");

const out = document.getElementById("readout");
setupHover(g, T0, T1, out, t => {
  const i = nearest(D.t, t);
  const f = (a, k) => a[k][i] === null ? "--" : a[k][i].toFixed(2);
  return "t=" + D.t[i].toFixed(3) + " s | F1 " + f(D.n4_khz, 0)
       + " / " + f(D.n6_khz, 0) + " kHz";
});
draw();
setupAudio(audio, g, T0, T1, draw, "#e63946");
"""


# ==========================================================================
# Demo 2 -- spectrogram + overlays (multipanel figure)
# ==========================================================================

SPEC_BODY = AUDIO_CONTROLS + """
<div class="card">
  <div class="controls">
    <button id="b_n4">Praat N=4</button>
    <button id="b_n6" class="on">Praat N=6</button>
    <button id="b_iaif" class="on">IAIF F1-F3</button>
    <button id="b_f0" class="on">F0 (IAIF)</button>
  </div>
  <canvas id="plot"></canvas>
  <p class="doc">Grey level: short-term spectrogram of the wahoo
  (0&ndash;3.5&thinsp;kHz). Overlays: Praat/Burg tracks interpolated on the
  analysis grid, IAIF vocal-tract formants, and the IAIF F0 track.
  Click to seek; the red cursor follows playback.</p>
</div>
"""

SPEC_JS = """
const D = __DATA__;
const g = makeCanvas("plot", 900, 420);
const audio = document.getElementById("audio");
const T0 = D.t[0], T1 = D.t[D.t.length - 1], V0 = 0, V1 = 3.5;
const show = { n4: false, n6: true, iaif: true, f0: true };

// Draw the spectrogram image from the embedded uint8 matrix.
// Image rows are written bottom-to-top so that row 0 (canvas top) is the
// highest frequency bin after drawImage.
const img = g.ctx.createImageData(D.spec[0].length, D.spec.length);
const nRows = D.spec.length, nCols = D.spec[0].length;
for (let j = 0; j < nRows; j++)
  for (let i = 0; i < nCols; i++) {
    const v = 255 - D.spec[j][i];          // dark = high energy
    const p = ((nRows - 1 - j) * nCols + i) * 4;
    img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
    img.data[p + 3] = 255;
  }

function draw() {
  drawFrame(g, T0, T1, V0, V1, "Time (s)", "Frequency (kHz)",
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 2, 3]);
  const ctx = g.ctx;
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, g.W - M.l - M.r, g.H - M.t - M.b); ctx.clip();
  // Off-screen canvas at native matrix size, then stretched.
  const off = document.createElement("canvas");
  off.width = D.spec[0].length; off.height = D.spec.length;
  off.getContext("2d").putImageData(img, 0, 0);
  const X0 = xScale.call(g, T0, T1)(T0), X1 = xScale.call(g, T0, T1)(T1);
  const Y0 = yScale.call(g, V0, V1)(V0), Y1 = yScale.call(g, V0, V1)(V1);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, X0, Y1, X1 - X0, Y0 - Y1);
  ctx.restore();
  if (show.f0)  plotLine(g, D.t, D.f0_khz, T0, T1, V0, V1, "#00b4d8", 2);
  if (show.n4)
    for (let k = 0; k < 3; k++)
      plotLine(g, D.t, D.n4_khz[k], T0, T1, V0, V1, "#ebb414", 1.6);
  if (show.n6)
    for (let k = 0; k < 3; k++)
      plotLine(g, D.t, D.n6_khz[k], T0, T1, V0, V1, "#8e44ad", 1.6);
  if (show.iaif) for (let k = 0; k < 3; k++)
    plotLine(g, D.t, D.iaif_khz[k], T0, T1, V0, V1, "#e63946", 1.4);
}
function bind(id, key) {
  document.getElementById(id).onclick = ev => {
    show[key] = !show[key];
    ev.target.classList.toggle("on", show[key]);
    draw();
  };
}
bind("b_n4", "n4"); bind("b_n6", "n6"); bind("b_iaif", "iaif"); bind("b_f0", "f0");

const out = document.getElementById("readout");
setupHover(g, T0, T1, out, t => {
  const i = nearest(D.t, t);
  const f = a => a[i] === null ? "--" : a[i].toFixed(2);
  return "t=" + D.t[i].toFixed(3) + " s | IAIF F1 " + f(D.iaif_khz[0])
       + " | F2 " + f(D.iaif_khz[1]) + " | F0 " + f(D.f0_khz) + " kHz";
});
draw();
setupAudio(audio, g, T0, T1, draw, "#e63946");
"""


# ==========================================================================
# Demo 3 -- F0 contours, filtered vs raw AC (all four wahoos)
# ==========================================================================

F0_BODY = """
<div class="card">
  <div class="controls">
    <select id="file"></select>
    <button id="play" class="playbtn">&#9654; Play</button>
    <button id="b_filt" class="on">filtered AC</button>
    <button id="b_raw" class="on">raw AC</button>
    <span class="badge">mean F0: <span id="means"></span> Hz</span>
    <span id="readout"></span>
  </div>
  <canvas id="plot"></canvas>
  <p class="doc">F0 contours estimated with Praat's autocorrelation
  method, with (blue) and without (orange) the Gaussian pre-filter that
  emulates recent Praat versions. On calls with subharmonics the two
  disagree &mdash; that gap is the nonlinear-phenomena diagnostic
  discussed in the article. Select a wahoo, click to seek, hover to
  read values.</p>
</div>
<audio id="audio" preload="auto"></audio>
"""

F0_JS = """
const FILES = __DATA__;   // [{name, audio, t, filt, raw}]
let cur = 0;
const audio = document.getElementById("audio");
const g = makeCanvas("plot", 900, 380);
const T0 = 0, T1 = 0.7, V0 = 100, V1 = 650;
const show = { filt: true, raw: true };

const sel = document.getElementById("file");
FILES.forEach((f, i) => {
  const o = document.createElement("option");
  o.value = i; o.textContent = f.name;
  sel.appendChild(o);
});
sel.onchange = () => select(parseInt(sel.value, 10));

function select(i) {
  cur = i;
  audio.src = FILES[i].audio;
  const f = FILES[i];
  document.getElementById("means").textContent =
    "filt " + mean(f.filt) + " / raw " + mean(f.raw);
  draw();
}
function mean(a) {
  const v = a.filter(x => x !== null);
  return (v.reduce((s, x) => s + x, 0) / v.length).toFixed(0);
}

function draw() {
  const f = FILES[cur];
  drawFrame(g, T0, T1, V0, V1, "Time (s)", "F0 (Hz)",
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [100, 200, 300, 400, 500, 600]);
  if (show.raw)  plotLine(g, f.t, f.raw, T0, T1, V0, V1, "#f4a261", 1.5);
  if (show.filt) plotLine(g, f.t, f.filt, T0, T1, V0, V1, "#0a6ebd", 2);
}
document.getElementById("b_filt").onclick = ev => {
  show.filt = !show.filt; ev.target.classList.toggle("on", show.filt); draw();
};
document.getElementById("b_raw").onclick = ev => {
  show.raw = !show.raw; ev.target.classList.toggle("on", show.raw); draw();
};

const out = document.getElementById("readout");
setupHover(g, T0, T1, out, t => {
  const f = FILES[cur], i = nearest(f.t, t);
  return "t=" + f.t[i].toFixed(3) + " s | filt " + (f.filt[i] || "--").toString()
       + " Hz | raw " + (f.raw[i] || "--").toString() + " Hz";
});
select(0);
setupAudio(audio, g, T0, T1, draw, "#e63946");
"""


# ==========================================================================
# Demo 4 -- glottal source: pulse, train, phase portrait
# ==========================================================================

GLOT_BODY = AUDIO_CONTROLS + """
<div class="card">
  <div class="controls">
    <button id="b_zoom">Reset zoom</button>
    <span class="badge">window: <span id="win">0.39 &ndash; 0.55 s</span></span>
    <span id="readout2"></span>
  </div>
  <canvas id="trainplot"></canvas>
  <div style="margin-top:10px">
    <canvas id="pulseplot"></canvas><canvas id="phaseplot"></canvas>
  </div>
  <p class="doc">Top: glottal pulse train reconstructed by IAIF
  overlap-add (zoom with the mouse wheel; drag not needed &mdash; hover
  to read amplitudes). Bottom left: single glottal cycle extracted at
  260&ndash;340&thinsp;ms. Bottom right: Hilbert phase portrait of the
  displayed window &mdash; a closed loop means periodic phonation; the
  spread visible here reflects the nonlinear (subharmonic / aperiodic)
  source dynamics discussed in the article.</p>
</div>
"""

GLOT_JS = """
const D = __DATA__;
const audio = document.getElementById("audio");
let win = { a: 0.39, b: 0.55 };
const gT = makeCanvas("trainplot", 900, 300);
const gP = makeCanvas("pulseplot", 440, 260);
const gH = makeCanvas("phaseplot", 440, 260);
gP.cv.style.width = "49%"; gP.cv.style.display = "inline-block";
gH.cv.style.width = "49%"; gH.cv.style.display = "inline-block";
gH.cv.style.marginLeft = "1.5%";

function drawTrain() {
  const i0 = Math.max(0, Math.floor(win.a * D.fs));
  const i1 = Math.min(D.train.length, Math.ceil(win.b * D.fs));
  const seg = D.train.slice(i0, i1);
  const ts = seg.map((_, k) => (i0 + k) / D.fs);
  drawFrame(gT, win.a, win.b, -1, 1, "Time (s)", "Amplitude (a.u.)",
            ticks(win.a, win.b, 5), [-1, 0, 1]);
  plotLine(gT, ts, seg, win.a, win.b, -1, 1, "#222", 0.8);
  drawPhase(ts, seg);
  document.getElementById("win").textContent =
    win.a.toFixed(2) + " \\u2013 " + win.b.toFixed(2) + " s";
}
function ticks(a, b, n) {
  const out = [];
  for (let k = 0; k <= n; k++) out.push(+(a + (b - a) * k / n).toFixed(2));
  return out;
}
// Phase portrait: the Hilbert transform is precomputed in Python for a
// grid of windows; pick the closest embedded window and rescale.
function drawPhase(ts, seg) {
  drawFrame(gH, -1.2, 1.2, -1.2, 1.2, "Real", "Imag", [-1, 0, 1], [-1, 0, 1]);
  let best = 0, bd = Infinity;
  for (let w = 0; w < D.phase_windows.length; w++) {
    const d = Math.abs(D.phase_windows[w] - (win.a + win.b) / 2);
    if (d < bd) { bd = d; best = w; }
  }
  const re = D.phase_re[best], im = D.phase_im[best];
  plotLine(gH, re, im, -1.2, 1.2, -1.2, 1.2, "#0a6ebd", 0.6);
}

function drawPulse() {
  const tms = D.pulse_ms;
  drawFrame(gP, 0, tms[tms.length - 1], -1, 1,
            "Time (ms)", "Amplitude (a.u.)", [0, 1, 2, 3], [-1, 0, 1]);
  plotLine(gP, tms, D.pulse, 0, tms[tms.length - 1], -1, 1, "#222", 1.4);
}

const out = document.getElementById("readout2");
setupHover(gT, win.a, win.b, out, t => {
  const i = Math.round(t * D.fs);
  return "t=" + t.toFixed(4) + " s | amp=" +
    (D.train[i] === undefined ? "--" : D.train[i].toFixed(3));
});
gT.cv.addEventListener("wheel", ev => {
  ev.preventDefault();
  const rect = gT.cv.getBoundingClientRect();
  const c = win.a + (ev.clientX - rect.left - M.l) / (gT.W - M.l - M.r) * (win.b - win.a);
  const f = ev.deltaY > 0 ? 1.15 : 1 / 1.15;
  const half = Math.min(0.3, Math.max(0.02, (win.b - win.a) / 2 * f));
  win = { a: Math.max(0, c - half), b: Math.min(D.dur, c + half) };
  drawTrain();
});
document.getElementById("b_zoom").onclick = () => {
  win = { a: 0.39, b: 0.55 }; drawTrain();
};
drawTrain(); drawPulse();
setupAudio(audio, gT, 0, D.dur, drawTrain, "#e63946");
"""


# ==========================================================================
# Data preparation
# ==========================================================================

def build_formants_data(data_dir, audio_uri):
    """Figure-1 data: Praat tracks N=4/N=6, merged, reference."""
    times4, F4 = extract_praat_formants(data_dir / "wahoo_4.wav", n_formants=4)
    times6, F6 = extract_praat_formants(data_dir / "wahoo_4.wav", n_formants=6)
    merged = np.concatenate(
        [F6[:4, :1], F4[:4, 1:9], F6[:4, 9:15]], axis=1)[:, :14]
    ref = np.genfromtxt(data_dir / "reference_formants_boe2026.csv",
                        delimiter=",", names=True)
    ref_khz = np.vstack([ref["F1_kHz"], ref["F2_kHz"],
                         ref["F3_kHz"], ref["F4_kHz"]])
    t_ref = np.linspace(times6[0], times6[13], 14)

    data = {
        "t": r2(times6[:15]),
        "n4_khz": r2(F4[:4, :15] / 1000.0),
        "n6_khz": r2(F6[:4, :15] / 1000.0),
        "t_merged": r2(times6[:14]),
        "merged_khz": r2(merged / 1000.0),
        "ref_t": r2(t_ref),
        "ref": r2(ref_khz),
    }
    vtl = {
        "n4": reby_mccomb_vtl(F4[:, :14]),
        "n6": reby_mccomb_vtl(F6[:4, :14]),
        "merged": reby_mccomb_vtl(merged),
    }
    badge = (f"VTL  N=4: {vtl['n4']:.1f} cm | N=6: {vtl['n6']:.1f} cm | "
             f"merged: {vtl['merged']:.1f} cm")
    return data, badge, audio_uri


def build_spectrogram_data(data_dir, audio_uri):
    """Multipanel data: spectrogram matrix + overlay tracks."""
    x, fs = load_audio(data_dir / "wahoo_4.wav")
    result = iaif_sliding_analysis(x, fs, n_formants=4)
    F6i = interpolate_praat_on_grid(data_dir / "wahoo_4.wav", 6, result.times)
    F4i = interpolate_praat_on_grid(data_dir / "wahoo_4.wav", 4, result.times)

    band = result.freqs <= 3500
    spec = quantize_db(result.spectrogram_db, band)
    data = {
        "fs": int(fs),
        "t": r2(result.times),
        "spec": spec.tolist(),
        "iaif_khz": r2(result.formants[:3] / 1000.0),
        "f0_khz": r2(result.f0_track / 1000.0),
        "n6_khz": r2(F6i[:3] / 1000.0),
        "n4_khz": r2(F4i[:3] / 1000.0),
    }
    badge = (f"sliding IAIF, {len(result.times)} frames x {spec.shape[0]} bins, "
             f"VTL(IAIF) = {reby_mccomb_vtl(result.formants):.1f} cm")
    return data, badge, audio_uri


def interpolate_praat_on_grid(audio_path, n_formants, target_times):
    """Praat tracks of ``n_formants`` interpolated on ``target_times`` (Hz)."""
    from baboon_acoustics import fill_leading_nan, interpolate_formants

    times, F = extract_praat_formants(audio_path, n_formants=n_formants)
    interp = interpolate_formants(times, F[:4], target_times)
    return np.array([fill_leading_nan(track) for track in interp])


def build_f0_data(data_dir):
    """F0 contours (filtered and raw AC) for the four wahoos."""
    files = []
    for i in (1, 2, 3, 4):
        path = data_dir / f"wahoo_{i}.wav"
        t_f, f0_f, _ = estimate_f0_praat(path)
        t_r, f0_r, _ = estimate_f0_praat(path, use_prefilter=False)
        # Common time grid in [0, 0.7] s; NaN outside voiced frames.
        grid = np.arange(0.0, 0.701, 0.01)
        filt = np.full(len(grid), np.nan)
        raw = np.full(len(grid), np.nan)
        for j, tg in enumerate(grid):
            if len(t_f) and np.min(np.abs(t_f - tg)) <= 0.006:
                filt[j] = f0_f[np.argmin(np.abs(t_f - tg))]
            if len(t_r) and np.min(np.abs(t_r - tg)) <= 0.006:
                raw[j] = f0_r[np.argmin(np.abs(t_r - tg))]
        files.append({
            "name": path.name,
            "audio": wav_data_uri(path),
            "t": r2(grid),
            "filt": [None if np.isnan(v) else round(float(v), 1) for v in filt],
            "raw": [None if np.isnan(v) else round(float(v), 1) for v in raw],
        })
    return files


def build_glottal_data(data_dir, audio_uri):
    """Glottal train, single pulse and precomputed phase portraits."""
    from scipy.signal import hilbert

    x, fs = load_audio(data_dir / "wahoo_4.wav", normalize=False)
    t_train, train = extract_glottal_train(x, fs)
    t_pulse, pulse = extract_glottal_pulse(x, fs, 0.26, 0.34)

    # Phase portraits precomputed for a grid of window centres (Hilbert
    # transform, amplitude-normalized, downsampled for embedding).
    centres = np.arange(0.06, 0.61, 0.03)
    half = 0.08
    phase_windows, phase_re, phase_im = [], [], []
    for c in centres:
        seg = train[int((c - half) * fs):int((c + half) * fs)]
        if len(seg) < 1024:
            continue
        seg = seg - np.mean(seg)
        seg = normalize_amplitude(seg, fs=fs, cutoff=50.0)
        analytic = hilbert(seg)
        scale = np.max(np.abs(analytic)) + 1e-12
        analytic = analytic / scale
        step = max(1, len(seg) // 1500)
        phase_windows.append(round(float(c), 3))
        phase_re.append(r3(analytic.real[::step]))
        phase_im.append(r3(analytic.imag[::step]))

    peak = np.max(np.abs(pulse))
    data = {
        "fs": int(fs),
        "dur": round(float(len(x) / fs), 4),
        "train": np.round(train / np.max(np.abs(train)), 3).tolist(),
        # Pulse time axis in ms (3 decimals: 0.1 µs resolution at 44.1 kHz
        # would be overkill, 2 decimals would collapse the 0-3 ms range).
        "pulse_ms": r3(t_pulse * 1000.0),
        "pulse": np.round(pulse / peak, 3).tolist(),
        "phase_windows": phase_windows,
        "phase_re": phase_re,
        "phase_im": phase_im,
    }
    badge = (f"pulse: {len(pulse)} samples "
             f"({len(pulse) / fs * 1000:.1f} ms), train: {len(train)} samples")
    return data, badge, audio_uri


def json_token(data):
    """Compact JSON dump for embedding (NaN -> null, compact separators)."""
    return json.dumps(_sanitize(data), allow_nan=False, separators=(",", ":"))


def _sanitize(obj):
    """Recursively replace NaN floats by None (JSON null)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ==========================================================================
# Page assembly
# ==========================================================================

def write_pages(output_dir):
    """Compute analyses and write the five demo pages."""
    data_dir = REPO_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    uri4 = wav_data_uri(data_dir / "wahoo_4.wav")

    # --- demo_formants ----------------------------------------------------
    data, badge, _ = build_formants_data(data_dir, uri4)
    body = (FORMANTS_BODY.replace("%%AUDIO%%", uri4)
            .replace("%%BADGE%%", badge))
    (output_dir / "demo_formants.html").write_text(
        page("Formant tracks", FORMANTS_DOC, body,
             FORMANTS_JS.replace("__DATA__", json_token(data))),
        encoding="utf-8")

    # --- demo_spectrogram ---------------------------------------------------
    data, badge, _ = build_spectrogram_data(data_dir, uri4)
    body = (SPEC_BODY.replace("%%AUDIO%%", uri4).replace("%%BADGE%%", badge))
    (output_dir / "demo_spectrogram.html").write_text(
        page("Spectrogram + formants", SPEC_DOC, body,
             SPEC_JS.replace("__DATA__", json_token(data))),
        encoding="utf-8")

    # --- demo_f0 ------------------------------------------------------------
    files = build_f0_data(data_dir)
    body = F0_BODY
    (output_dir / "demo_f0.html").write_text(
        page("F0 contours", F0_DOC, body,
             F0_JS.replace("__DATA__", json_token(files))),
        encoding="utf-8")

    # --- demo_glottal -------------------------------------------------------
    data, badge, _ = build_glottal_data(data_dir, uri4)
    body = (GLOT_BODY.replace("%%AUDIO%%", uri4).replace("%%BADGE%%", badge))
    (output_dir / "demo_glottal.html").write_text(
        page("Glottal source", GLOT_DOC, body,
             GLOT_JS.replace("__DATA__", json_token(data))),
        encoding="utf-8")

    # --- index ---------------------------------------------------------------
    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    for p in sorted(output_dir.glob("*.html")):
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")


FORMANTS_DOC = ("Praat/Burg formant tracks of the wahoo call, with the "
                "number of LPC poles N as the interactive knob: toggle "
                "N=4, N=6 and the merged trajectory of the article against "
                "the digitized reference, while listening to the call.")
SPEC_DOC = ("Spectrogram of the wahoo with the Praat/Burg tracks, the IAIF "
            "vocal-tract formants and the IAIF F0 overlaid; playback is "
            "synchronized with a cursor and clicking seeks the audio.")
F0_DOC = ("F0 contours of the four wahoo calls, comparing Praat's "
          "autocorrelation with and without the Gaussian pre-filter that "
          "emulates recent Praat versions &mdash; the gaps between the two "
          "curves reveal subharmonic (nonlinear) source dynamics.")
GLOT_DOC = ("Glottal source recovered by IAIF inverse filtering: full pulse "
            "train (zoom with the mouse wheel), single glottal cycle, and "
            "Hilbert phase portrait of the displayed window.")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>baboon-acoustics - interactive demos</title>
<style>
""" + COMMON_CSS + """
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
.grid .card a { color: var(--accent); font-weight:bold; text-decoration:none; font-size:16px; }
</style>
</head>
<body>
<h1>baboon-acoustics &mdash; interactive demos</h1>
<p class="doc">Self-contained HTML demos of the analyses of the article:
each page embeds the audio (base64) and the precomputed analysis data, so
it runs offline in any browser. Plots are interactive: toggles, hover
readouts, playback cursor, click-to-seek, mouse-wheel zoom. Regenerate
with <code>python scripts/make_demos.py</code>.</p>
<div class="grid">
  <div class="card"><a href="demo_formants.html">Formant tracks</a>
    <p class="doc">Praat/Burg N=4 vs N=6 vs merged trajectories against the
    reference, with VTL estimates and synchronized audio playback.</p></div>
  <div class="card"><a href="demo_spectrogram.html">Spectrogram + formants</a>
    <p class="doc">Short-term spectrogram with Praat, IAIF and F0 overlays;
    playback cursor and seek.</p></div>
  <div class="card"><a href="demo_f0.html">F0 contours</a>
    <p class="doc">Filtered vs raw autocorrelation F0 for the four wahoos;
    the disagreements flag subharmonic dynamics.</p></div>
  <div class="card"><a href="demo_glottal.html">Glottal source</a>
    <p class="doc">IAIF pulse train (wheel zoom), single glottal cycle and
    Hilbert phase portrait.</p></div>
</div>
<p class="doc">Part of the <code>baboon-acoustics</code> repository
(supplementary material of the article "On the Reliability of Wahoo
Formant Measurements: A Response to Bo&euml; et al. (2026)", F.
Berthommier, 2026). See the repository README for the Python API.</p>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "demos"),
                        help="output directory (default: demos/)")
    args = parser.parse_args()
    write_pages(Path(args.output_dir))


if __name__ == "__main__":
    main()

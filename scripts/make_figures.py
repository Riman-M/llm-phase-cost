#!/usr/bin/env python3
"""
make_figures.py -- the four figures, from measured values.

Values are transcribed from the measurement outputs rather than recomputed,
so the figures cannot silently drift from the numbers in the tables:

  joint_torch.csv / joint_llama.csv   single Kaggle session, Xeon 2.20 GHz,
                                      2 physical cores, 4 threads
  cost_model_{f32,f16,q4}.csv         cost model over the four Azure traces
  trace_stats.csv                     workload characterisation

Outputs PDF (vector, for LaTeX) into ./figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def tidy(ax, grid="y"):
    """Remove chartjunk: top and right spines, and grid on one axis only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid, linestyle="-")
    ax.grid(axis="x" if grid == "y" else "y", visible=False)
    ax.tick_params(length=2.5)


BLUE, ORANGE, GREEN, GREY = "#2f5d8a", "#c26a3d", "#4a7c59", "#8a8a8a"
LIGHT = "#9fb8cd"

# --------------------------------------------------------------------------
# Figure 1 -- the workload shift
# --------------------------------------------------------------------------
traces = ["2023\nconv", "2024\nconv", "2023\ncode", "2024\ncode"]
ctx_mean = [1154.7, 1631.6, 2047.8, 2511.3]
gen_mean = [211.1, 105.5, 27.9, 22.7]
ratio = [5.47, 15.46, 73.45, 110.68]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3))
x = np.arange(4)
a.bar(x - 0.19, ctx_mean, 0.38, label="context (prefill)", color=BLUE)
a.bar(x + 0.19, gen_mean, 0.38, label="generated (decode)", color=ORANGE)
a.set_xticks(x)
a.set_xticklabels(traces)
a.set_ylabel("mean tokens per request")
a.legend(loc="upper left", fontsize=7)
a.set_title("(a) token counts", fontsize=8, loc="left", color="#222222")
tidy(a)

b.bar(x, ratio, 0.5, color=BLUE)
for i, v in enumerate(ratio):
    b.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=7)
b.set_xticks(x)
b.set_xticklabels(traces)
b.set_ylabel("prefill : decode token ratio")
b.set_ylim(0, 128)
b.set_title("(b) ratio widens in both services", fontsize=8, loc="left",
            color="#222222")
tidy(b)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_workload_shift.pdf")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2 -- decode is a weight read at memory bandwidth
#
# The point of the figure: two frameworks whose decode latency differs 2x
# land on the same bandwidth once byte counts are divided out.
# --------------------------------------------------------------------------
cfg = ["PyTorch\nfloat32", "llama.cpp\nf16", "llama.cpp\nQ4_K_M"]
weights_gb = [6.18, 3.09, 0.86]
ms_tok = [336.4, 165.9, 73.4]
bw = [g / (m / 1000) for g, m in zip(weights_gb, ms_tok)]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3))
x = np.arange(3)
a.bar(x, ms_tok, 0.5, color=[BLUE, ORANGE, GREEN])
for i, v in enumerate(ms_tok):
    a.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=7)
a.set_xticks(x)
a.set_xticklabels(cfg)
a.set_ylabel("decode latency (ms/token)")
a.set_ylim(0, 400)
a.set_title("(a) latency differs by $2\\times$", fontsize=8, loc="left",
            color="#222222")
tidy(a)

b.bar(x, bw, 0.5, color=[BLUE, ORANGE, GREEN])
for i, v in enumerate(bw):
    b.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
b.axhline(18.5, ls="--", lw=0.8, color=GREY)
b.text(2.42, 18.9, "bandwidth\nceiling", fontsize=6.5, color=GREY, ha="right")
b.set_xticks(x)
b.set_xticklabels(cfg)
b.set_ylabel("effective bandwidth (GB/s)")
b.set_ylim(0, 23)
b.set_title("(b) bandwidth does not", fontsize=8, loc="left", color="#222222")
tidy(b)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_bandwidth.pdf")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 3 -- phase cost gap by precision and context
# --------------------------------------------------------------------------
ctxs = [128, 512, 1024, 2048, 4096, 8192]
gap_f32 = [10.6, 11.6, 12.1, 12.4, 13.2, 15.1]
gap_f16 = [3.6, 3.6, 3.6, 3.6, 3.5, 3.8]
gap_q4 = [2.6, 2.8, 2.8, 3.0, 3.1, 3.3]

fig, ax = plt.subplots(figsize=(3.3, 2.4))
for g, lbl, c, mk in ((gap_f32, "float32", BLUE, "o"),
                      (gap_f16, "f16", ORANGE, "s"),
                      (gap_q4, "Q4_K_M", GREEN, "^")):
    ax.plot(ctxs, g, marker=mk, ms=3.2, lw=1.1, color=c, label=lbl)
ax.set_xscale("log", base=2)
ax.set_xticks(ctxs)
ax.set_xticklabels([str(c) for c in ctxs])
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("decode cost / prefill cost\n(per token)")
ax.set_ylim(0, 17)
for g, lbl, c, dy in ((gap_f32, "float32", BLUE, 0),
                      (gap_f16, "f16", ORANGE, 5),
                      (gap_q4, "Q4_K_M", GREEN, -6)):
    ax.annotate(lbl, (ctxs[-1], g[-1]), textcoords="offset points",
                xytext=(5, dy), fontsize=7, color=c, va="center")
ax.set_xlim(100, 14000)
tidy(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_phase_gap.pdf")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 4 -- the headline: quantisation return collapses with prefill share
# --------------------------------------------------------------------------
names = ["2023 conv", "2024 conv", "2023 code", "2024 code"]
prefill_share_f32 = [32.4, 55.6, 86.2, 90.2]
q4_speedup = [2.13, 1.49, 1.06, 1.01]
f16_rel = [1.16, 0.88, 0.66, 0.64]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))

a.plot(prefill_share_f32, q4_speedup, "o-", ms=5, lw=1.3, color=BLUE)
offsets = [(6, 4), (6, 4), (-6, 11), (5, -2)]
for xx, yy, nn, off in zip(prefill_share_f32, q4_speedup, names, offsets):
    a.annotate(nn, (xx, yy), textcoords="offset points",
               xytext=off, fontsize=6.5, color="#222222")
a.axhline(1.0, ls=":", lw=0.8, color=GREY)
a.set_xlabel("prefill share of service time at float32 (%)")
a.set_ylabel("speedup from 4-bit quantisation")
a.set_ylim(0.9, 2.4)
a.set_xlim(25, 108)
a.text(30, 1.04, "no benefit", fontsize=6.5, color=GREY, ha="left")
a.set_title("(a) the return collapses with prefill share", fontsize=8,
            loc="left", color="#222222")
tidy(a)

x = np.arange(4)
cols = [GREEN if v >= 1 else ORANGE for v in f16_rel]
b.bar(x, f16_rel, 0.5, color=cols)
b.axhline(1.0, ls="-", lw=0.8, color="black")
for i, v in enumerate(f16_rel):
    b.text(i, v + 0.03 if v >= 1 else v - 0.09, f"{v:.2f}",
           ha="center", fontsize=7)
b.set_xticks(x)
b.set_xticklabels([n.replace(" ", "\n") for n in names])
b.set_ylabel("f16 speedup vs float32")
b.set_ylim(0, 1.35)
b.text(3.45, 1.04, "break-even", fontsize=6.5, color="#222222", ha="right")
b.set_title("(b) below the line means slower", fontsize=8, loc="left",
            color="#222222")
tidy(b)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_quantisation_return.pdf")
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}")

#!/usr/bin/env python3
"""
trace_prep.py -- acquisition and characterisation of open LLM inference traces.

Sources (all CC-BY, Azure/AzurePublicDataset):
  2023: AzureLLMInferenceTrace_{conv,code}.csv   (Splitwise, ISCA 2024)
  2024: AzureLLMInferenceTrace_{conv,code}_1week.csv (DynamoLLM, HPCA 2025)

Schema for every file: TIMESTAMP, ContextTokens, GeneratedTokens.
There are no session identifiers. Do not claim prefix-reuse results from these.

Outputs (into --outdir):
  trace_stats.csv       one row per trace, all characterisation numbers
  arrivals_<tag>.csv    per-second arrival counts (for the burstiness figure)
  fig_shift.pdf         prefill/decode shift 2023 -> 2024
  fig_arrivals.pdf      arrival burstiness, conv vs code

Runs on CPU only. The 2024 files are ~1 GB each and are streamed in chunks,
so peak memory stays near 1 GB regardless of file size.

Usage:
  python3 trace_prep.py --download --outdir results
  python3 trace_prep.py --outdir results --skip 2024_code   # if disk is tight
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
REL = ("https://github.com/Azure/AzurePublicDataset/releases/download"
       "/dataset-llm-2024")

TRACES = {
    "2023_conv": f"{RAW}/AzureLLMInferenceTrace_conv.csv",
    "2023_code": f"{RAW}/AzureLLMInferenceTrace_code.csv",
    "2024_conv": f"{REL}/AzureLLMInferenceTrace_conv_1week.csv",
    "2024_code": f"{REL}/AzureLLMInferenceTrace_code_1week.csv",
}

CHUNK = 2_000_000          # rows per chunk when streaming
PCTS = [50, 90, 95, 99]


def fetch(tag, url, datadir):
    path = os.path.join(datadir, f"{tag}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  [{tag}] already present ({os.path.getsize(path)/1e6:.0f} MB)")
        return path
    print(f"  [{tag}] downloading ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"  [{tag}] done ({os.path.getsize(path)/1e6:.0f} MB)")
    return path


class Accumulator:
    """Streaming accumulator so the 1 GB traces never land in memory at once.

    Token distributions are held as integer histograms, which is exact for
    percentiles (token counts are small non-negative integers) and costs a
    fixed ~200 KB regardless of how many requests we read.
    """

    MAXTOK = 200_000

    def __init__(self):
        self.n = 0
        self.ctx_hist = np.zeros(self.MAXTOK + 1, dtype=np.int64)
        self.gen_hist = np.zeros(self.MAXTOK + 1, dtype=np.int64)
        self.ctx_sum = 0
        self.gen_sum = 0
        self.tmin = None
        self.tmax = None
        self.per_sec = {}

    def add(self, df):
        ctx = df["ContextTokens"].to_numpy(dtype=np.int64, copy=False)
        gen = df["GeneratedTokens"].to_numpy(dtype=np.int64, copy=False)
        np.add.at(self.ctx_hist, np.clip(ctx, 0, self.MAXTOK), 1)
        np.add.at(self.gen_hist, np.clip(gen, 0, self.MAXTOK), 1)
        self.ctx_sum += int(ctx.sum())
        self.gen_sum += int(gen.sum())
        self.n += len(df)

        t = pd.to_datetime(df["TIMESTAMP"], format="mixed", utc=True)
        lo, hi = t.min(), t.max()
        self.tmin = lo if self.tmin is None else min(self.tmin, lo)
        self.tmax = hi if self.tmax is None else max(self.tmax, hi)
        for sec, cnt in t.dt.floor("s").value_counts().items():
            self.per_sec[sec] = self.per_sec.get(sec, 0) + int(cnt)

    @staticmethod
    def _pct(hist, total, p):
        target = total * p / 100.0
        return int(np.searchsorted(np.cumsum(hist), target))

    def arrivals(self):
        """Per-second arrival counts, zero-filled across the whole span."""
        s = pd.Series(self.per_sec).sort_index()
        idx = pd.date_range(s.index.min(), s.index.max(), freq="s")
        return s.reindex(idx, fill_value=0)

    def summary(self, tag):
        span = (self.tmax - self.tmin).total_seconds()
        arr = self.arrivals()
        row = {
            "trace": tag,
            "requests": self.n,
            "span_hours": round(span / 3600, 2),
            "mean_rate_rps": round(self.n / span, 2) if span else float("nan"),
            "ctx_mean": round(self.ctx_sum / self.n, 1),
            "gen_mean": round(self.gen_sum / self.n, 1),
            # The headline number: token-weighted prefill:decode ratio.
            "prefill_decode_ratio": round(self.ctx_sum / max(self.gen_sum, 1), 2),
            "arr_mean_rps": round(float(arr.mean()), 2),
            "arr_p99_rps": int(np.percentile(arr, 99)),
            "arr_peak_rps": int(arr.max()),
            "arr_cv": round(float(arr.std() / arr.mean()), 3),
        }
        for p in PCTS:
            row[f"ctx_p{p}"] = self._pct(self.ctx_hist, self.n, p)
            row[f"gen_p{p}"] = self._pct(self.gen_hist, self.n, p)
        return row


def process(tag, path):
    acc = Accumulator()
    for chunk in pd.read_csv(
        path, chunksize=CHUNK,
        usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
    ):
        acc.add(chunk)
        print(f"    [{tag}] {acc.n:,} requests", end="\r", flush=True)
    print(f"    [{tag}] {acc.n:,} requests            ")
    return acc


def figures(stats, arrivals, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3})

    # Figure 1: the prefill/decode shift.
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    tags = list(stats["trace"])
    x = np.arange(len(tags))
    a.bar(x - 0.2, stats["ctx_p50"], 0.4, label="context (prefill)",
          color="#33608c")
    a.bar(x + 0.2, stats["gen_p50"], 0.4, label="generated (decode)",
          color="#c4702a")
    a.set_xticks(x)
    a.set_xticklabels(tags, rotation=20, ha="right")
    a.set_ylabel("median tokens per request")
    a.legend(frameon=False)

    b.bar(x, stats["prefill_decode_ratio"], 0.5, color="#33608c")
    for i, v in enumerate(stats["prefill_decode_ratio"]):
        b.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels(tags, rotation=20, ha="right")
    b.set_ylabel("prefill : decode token ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_shift.pdf"))
    plt.close(fig)

    # Figure 2: arrival burstiness. Normalised so shapes are comparable.
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for tag, arr in arrivals.items():
        win = arr.iloc[: 30 * 60]                    # first 30 minutes
        ax.plot(np.arange(len(win)) / 60.0, win / max(arr.mean(), 1e-9),
                lw=0.7, label=tag)
    ax.set_xlabel("minutes")
    ax.set_ylabel("arrivals / mean")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_arrivals.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--download", action="store_true",
                    help="fetch any trace not already on disk")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="trace tags to leave out (e.g. 2024_code)")
    args = ap.parse_args()

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    rows, arrivals = [], {}
    for tag, url in TRACES.items():
        if tag in args.skip:
            print(f"[{tag}] skipped")
            continue
        path = os.path.join(args.datadir, f"{tag}.csv")
        if args.download:
            path = fetch(tag, url, args.datadir)
        if not os.path.exists(path):
            print(f"[{tag}] missing; rerun with --download", file=sys.stderr)
            continue
        acc = process(tag, path)
        rows.append(acc.summary(tag))
        arr = acc.arrivals()
        arrivals[tag] = arr
        arr.to_csv(os.path.join(args.outdir, f"arrivals_{tag}.csv"),
                   header=["requests"])

    if not rows:
        sys.exit("no traces processed")

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.outdir, "trace_stats.csv"), index=False)
    print("\n" + stats.to_string(index=False))
    figures(stats, arrivals, args.outdir)
    print(f"\nwrote {args.outdir}/trace_stats.csv, fig_shift.pdf, "
          f"fig_arrivals.pdf")


if __name__ == "__main__":
    main()

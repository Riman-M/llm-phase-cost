#!/usr/bin/env python3
"""
ratio_stability.py -- does the prefill:decode ratio hold across hours?

Why this exists. The headline claim compares a ONE HOUR sample from November
2023 against a ONE WEEK sample from May 2024. Request volume in the 2024 week
swings 1.78x (conv) and 8.96x (code) by hour of day. If the token mix also
varies by hour, part of the apparent shift could be a sampling artifact rather
than a real change, and that is the first thing a reviewer will test.

Two checks, both on data we already hold:

  1. Hourly stability. Compute the token-weighted prefill:decode ratio for
     each of the 168 hours in the 2024 week. If the ratio is roughly flat
     while volume swings by 9x, then WHICH hour you sample barely matters,
     and the 2023 one-hour window is a fair baseline. If it swings, the
     comparison has to be hour-matched and the claim weakened accordingly.

  2. Hour-matched comparison. Pull from the 2024 week only the hours whose
     hour-of-day matches the 2023 sample window, and recompute the ratio.
     This is the like-for-like number to put in the paper alongside the
     all-hours one.

Both 2023 files are small enough to load whole; 2024 is streamed.

Usage:
  python3 ratio_stability.py --datadir data --outdir results
"""

import argparse
import os

import numpy as np
import pandas as pd

CHUNK = 2_000_000
COLS = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]


def hourly_sums(path, chunksize=CHUNK):
    """Streaming per-hour token sums and request counts."""
    ctx = {}
    gen = {}
    cnt = {}
    for ch in pd.read_csv(path, usecols=COLS, chunksize=chunksize):
        t = pd.to_datetime(ch["TIMESTAMP"], format="mixed", utc=True)
        hr = t.dt.floor("h")
        g = ch.assign(_hr=hr).groupby("_hr")
        for k, v in g["ContextTokens"].sum().items():
            ctx[k] = ctx.get(k, 0) + int(v)
        for k, v in g["GeneratedTokens"].sum().items():
            gen[k] = gen.get(k, 0) + int(v)
        for k, v in g.size().items():
            cnt[k] = cnt.get(k, 0) + int(v)
    df = pd.DataFrame({"ctx": pd.Series(ctx), "gen": pd.Series(gen),
                       "requests": pd.Series(cnt)}).sort_index()
    df["ratio"] = df["ctx"] / df["gen"].clip(lower=1)
    df["hour_of_day"] = df.index.hour
    return df


def describe(tag, df):
    r = df["ratio"]
    print(f"\n=== {tag} ===")
    print(f"  hours                {len(df)}")
    print(f"  requests             {df['requests'].sum():,}")
    print(f"  ratio  median        {r.median():.2f}")
    print(f"  ratio  p5 - p95      {np.percentile(r, 5):.2f} - "
          f"{np.percentile(r, 95):.2f}")
    print(f"  ratio  min - max     {r.min():.2f} - {r.max():.2f}")
    print(f"  ratio  CV            {r.std()/r.mean():.3f}")
    print(f"  volume swing         {df['requests'].max()/df['requests'].min():.2f}x")
    # The decisive number: if the ratio barely moves while volume swings,
    # the one-hour 2023 baseline is defensible.
    print(f"  ratio spread / volume spread  "
          f"{(r.max()/r.min()) / (df['requests'].max()/df['requests'].min()):.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    out = {}
    for service in ("conv", "code"):
        p23 = os.path.join(args.datadir, f"2023_{service}.csv")
        p24 = os.path.join(args.datadir, f"2024_{service}.csv")
        if not (os.path.exists(p23) and os.path.exists(p24)):
            print(f"[{service}] missing input, skipped")
            continue

        d23 = hourly_sums(p23)
        d24 = hourly_sums(p24)
        describe(f"2023 {service}", d23)
        describe(f"2024 {service}", d24)

        # Hour-matched: restrict 2024 to the hours-of-day the 2023 sample
        # actually covers, then recompute on pooled tokens.
        hods = sorted(set(d23["hour_of_day"]))
        m = d24[d24["hour_of_day"].isin(hods)]
        matched = m["ctx"].sum() / max(m["gen"].sum(), 1)
        allhours = d24["ctx"].sum() / max(d24["gen"].sum(), 1)
        base23 = d23["ctx"].sum() / max(d23["gen"].sum(), 1)

        print(f"\n  -- {service}: like-for-like --")
        print(f"  2023 sample hours-of-day  {hods}")
        print(f"  2023 ratio                          {base23:.2f}")
        print(f"  2024 ratio, all 168 hours           {allhours:.2f}")
        print(f"  2024 ratio, matched hours only      {matched:.2f}")
        print(f"  shift, all hours                    {allhours/base23:.2f}x")
        print(f"  shift, hour-matched                 {matched/base23:.2f}x")

        d24.to_csv(os.path.join(args.outdir, f"hourly_2024_{service}.csv"))
        out[service] = (d23, d24, base23, allhours, matched)

    if not out:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3})

    fig, axes = plt.subplots(1, len(out), figsize=(3.5 * len(out), 2.8),
                             squeeze=False)
    for ax, (service, (d23, d24, b23, allh, match)) in zip(axes[0],
                                                           out.items()):
        h = np.arange(len(d24))
        ax.plot(h, d24["ratio"], lw=0.8, color="#33608c",
                label="2024, per hour")
        ax.axhline(allh, ls="--", lw=1, color="#33608c",
                   label=f"2024 pooled {allh:.1f}")
        ax.axhline(b23, ls=":", lw=1.2, color="#c4702a",
                   label=f"2023 sample {b23:.1f}")
        ax.set_title(service)
        ax.set_xlabel("hour of the 2024 week")
        ax.set_ylabel("prefill : decode")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_ratio_stability.pdf"))
    print(f"\nwrote {args.outdir}/fig_ratio_stability.pdf and hourly_*.csv")


if __name__ == "__main__":
    main()

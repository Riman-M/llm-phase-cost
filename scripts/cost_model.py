#!/usr/bin/env python3
"""
cost_model.py -- from measured phase costs to provisioning error.

The argument in three steps.

  1. Fit cost curves to the profiler output. Prefill and decode get different
     functional forms because they are bounded by different resources:

         prefill_ms(c) = a*c + b*c^2      linear GEMM term, quadratic attention
         decode_ms(c)  = a + b*c          fixed per-token overhead, plus a KV
                                          cache read growing with context

     Both are fitted per model and per thread count, with R^2 reported. A poor
     fit is a signal not to trust the extrapolation, so it is printed rather
     than hidden.

  2. Apply them to REAL request distributions from the Azure traces, one
     request at a time, rather than to the mean request. Service time is
     non-linear in context, so the mean request is not the average cost, and
     the gap grows with how heavy the context tail is.

  3. Compare against a planner that treats capacity as proportional to token
     count, which is what you get from reading token ratios off a workload
     characterisation. Calibrate it on one workload, apply it to another,
     and report how far wrong it lands.

Capacity numbers use a single-replica service-time model. This is a
first-order planning estimate, not a simulation of a batching server: real
engines overlap prefill and decode across requests. The RATIO between
planners is the claim, and it is insensitive to that, because both planners
are handed the same server model.

Usage:
  python3 cost_model.py --profile cpu_profile_full.csv --traces data/*.csv
  python3 cost_model.py --profile cpu_profile_full.csv --traces data/2024_code.csv \
                        --model Qwen/Qwen2.5-1.5B --threads 4
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

CHUNK = 2_000_000
REF = "2023_conv"          # the workload the older characterisations describe


# --------------------------------------------------------------------------
# Cost curves
# --------------------------------------------------------------------------

def fit_curves(profile, model, threads):
    """Least-squares fits of both phases, with R^2 for each."""
    d = profile[(profile.model == model) & (profile.threads == threads)]
    out = {}

    p = d[d.phase == "prefill"].sort_values("context_tokens")
    c, y = p.context_tokens.to_numpy(float), p.median_ms.to_numpy(float)
    A = np.column_stack([c, c ** 2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    out["prefill"] = (coef, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    q = d[d.phase == "decode"].sort_values("context_tokens")
    c2, y2 = q.context_tokens.to_numpy(float), q.median_ms.to_numpy(float)
    A2 = np.column_stack([np.ones_like(c2), c2])
    coef2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    pred2 = A2 @ coef2
    out["decode"] = (coef2,
                     1 - ((y2 - pred2) ** 2).sum() / ((y2 - y2.mean()) ** 2).sum())
    out["ctx_range"] = (c.min(), c.max())
    return out


def prefill_ms(curves, ctx):
    a, b = curves["prefill"][0]
    return a * ctx + b * ctx ** 2


def decode_ms(curves, ctx):
    a, b = curves["decode"][0]
    return a + b * ctx


def service_time_ms(curves, ctx, gen):
    return prefill_ms(curves, ctx) + gen * decode_ms(curves, ctx)


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------

def trace_costs(path, curves, cap):
    """Stream a trace, accumulating per-request service time.

    Contexts beyond the profiled range are clipped rather than extrapolated,
    and the fraction clipped is reported so the reader can judge it.
    """
    n = tot_pf = tot_dc = tot_ctx = tot_gen = 0
    clipped = 0
    for ch in pd.read_csv(path, usecols=["ContextTokens", "GeneratedTokens"],
                          chunksize=CHUNK):
        ctx = ch.ContextTokens.to_numpy(float)
        gen = ch.GeneratedTokens.to_numpy(float)
        clipped += int((ctx > cap).sum())
        ctxc = np.clip(ctx, 1, cap)
        pf = prefill_ms(curves, ctxc)
        dc = gen * decode_ms(curves, ctxc)
        tot_pf += pf.sum()
        tot_dc += dc.sum()
        tot_ctx += ctx.sum()
        tot_gen += gen.sum()
        n += len(ch)
    return {
        "requests": n,
        "mean_service_s": (tot_pf + tot_dc) / n / 1000,
        "mean_prefill_s": tot_pf / n / 1000,
        "mean_decode_s": tot_dc / n / 1000,
        "prefill_share": tot_pf / (tot_pf + tot_dc),
        "mean_tokens": (tot_ctx + tot_gen) / n,
        "token_ratio": tot_ctx / max(tot_gen, 1),
        "clipped_frac": clipped / n,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu_profile_full.csv")
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arrivals", default=None,
                    help="directory holding arrivals_<tag>.csv for peak sizing")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prof = pd.read_csv(args.profile)
    prof = prof[prof.phase.isin(["prefill", "decode"])]
    curves = fit_curves(prof, args.model, args.threads)

    (pa, pb), pr2 = curves["prefill"]
    (da, db), dr2 = curves["decode"]
    lo, hi = curves["ctx_range"]
    print(f"model {args.model}, {args.threads} threads, "
          f"profiled context {lo:.0f}-{hi:.0f}")
    print(f"  prefill_ms(c) = {pa:.4f}*c + {pb:.3e}*c^2      R2 = {pr2:.4f}")
    print(f"  decode_ms(c)  = {da:.3f} + {db:.3e}*c          R2 = {dr2:.4f}")
    print(f"  fixed decode overhead is {da:.1f} ms/token, "
          f"{100*da/decode_ms(curves, 1024):.0f}% of the cost at ctx=1024")

    paths = []
    for pat in args.traces:
        paths.extend(sorted(glob.glob(pat)))
    rows = []
    for p in paths:
        tag = os.path.splitext(os.path.basename(p))[0]
        r = trace_costs(p, curves, hi)
        r["trace"] = tag
        rows.append(r)
        print(f"  [{tag}] {r['requests']:,} requests, "
              f"{100*r['clipped_frac']:.2f}% clipped at ctx={hi:.0f}")

    df = pd.DataFrame(rows).set_index("trace")
    df["throughput_rps"] = 1.0 / df.mean_service_s
    df["ms_per_token"] = df.mean_service_s * 1000 / df.mean_tokens

    print("\n" + "=" * 78)
    print("PER-REQUEST COST (over the real request distribution)")
    print(f"{'trace':<12}{'tok ratio':>10}{'service s':>11}{'prefill %':>11}"
          f"{'req/s/node':>12}{'ms/token':>10}")
    for t, r in df.iterrows():
        print(f"{t:<12}{r.token_ratio:>10.1f}{r.mean_service_s:>11.2f}"
              f"{100*r.prefill_share:>10.1f}%{r.throughput_rps:>12.4f}"
              f"{r.ms_per_token:>10.2f}")

    if REF in df.index:
        print("\n" + "=" * 78)
        print(f"PROVISIONING ERROR of a token-proportional planner")
        print(f"calibrated on {REF}, applied elsewhere")
        ref_cost_per_token = df.loc[REF, "ms_per_token"]
        print(f"{'trace':<12}{'predicted s':>13}{'actual s':>11}"
              f"{'error':>10}   verdict")
        for t, r in df.iterrows():
            pred = ref_cost_per_token * r.mean_tokens / 1000
            err = pred / r.mean_service_s
            verdict = ("over-provisions by "
                       f"{err:.2f}x" if err > 1.05 else
                       "under-provisions by "
                       f"{1/err:.2f}x" if err < 0.95 else "close enough")
            print(f"{t:<12}{pred:>13.2f}{r.mean_service_s:>11.2f}"
                  f"{err:>10.2f}x   {verdict}")

    if args.arrivals:
        print("\n" + "=" * 78)
        print("CAPACITY AT OBSERVED ARRIVAL RATES")
        print(f"{'trace':<12}{'mean rps':>10}{'p99 rps':>10}"
              f"{'nodes@mean':>12}{'nodes@p99':>11}{'headroom':>10}")
        for t, r in df.iterrows():
            f = os.path.join(args.arrivals, f"arrivals_{t}.csv")
            if not os.path.exists(f):
                continue
            a = pd.read_csv(f, index_col=0).iloc[:, 0]
            mean_n = a.mean() * r.mean_service_s
            p99_n = np.percentile(a, 99) * r.mean_service_s
            print(f"{t:<12}{a.mean():>10.1f}{np.percentile(a,99):>10.0f}"
                  f"{mean_n:>12.0f}{p99_n:>11.0f}{p99_n/mean_n:>9.2f}x")

    df.to_csv(os.path.join(args.outdir, "cost_model.csv"))
    print(f"\nwrote {args.outdir}/cost_model.csv")


if __name__ == "__main__":
    main()

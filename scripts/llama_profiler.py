#!/usr/bin/env python3
"""
llama_profiler.py -- prefill/decode profiling under an optimised runtime.

WHY THIS EXISTS

The PyTorch measurement found decode cost to be 93% fixed per-token overhead
(305 ms of a 320 ms step at 1024 tokens). That constant is eager-mode float32
execution, not a property of autoregressive decoding, and it inflates the
measured prefill/decode gap. This script re-measures the same two phases under
llama.cpp, which fuses kernels and supports quantised weights.

Two confounds are separated deliberately:

  f16     same precision as the PyTorch run (near enough), so the difference
          is attributable to the RUNTIME alone.
  q8_0    8-bit, the mild quantisation case.
  q4_k_m  4-bit, the deployment-realistic case.

Reading the result: if the phase gap holds at f16, the asymmetry is structural
and the paper's claim survives. If it collapses, the 13x figure was largely
framework cost and the provisioning error must be restated downward.

METHOD -- deliberately identical to cpu_profiler.py

  prefill : one pass over C tokens, KV cache empty beforehand.
  decode  : per-token latency of single-token steps with a KV cache warm from
            a C-token prompt; the seeding pass is excluded from the timing.

Random in-vocabulary token ids, two warmup passes discarded, median of
repetitions, cell order shuffled, a reference cell re-measured periodically to
expose drift. Same output columns as cpu_profiler.py, with the quantisation
folded into the model name, so cost_model.py consumes this file unchanged.

SETUP

  pip install llama-cpp-python huggingface_hub

  Building llama-cpp-python from source takes 5-10 minutes. If a prebuilt
  wheel is offered for the platform it will be used instead.

  python llama_profiler.py --out llama_profile.csv

MUST RUN ON THE SAME HOST as cpu_profile_merged.csv, or the comparison is
between two machines rather than two runtimes.
"""

import argparse
import csv
import os
import platform
import random
import signal
import statistics
import sys
import time

# (HuggingFace repo, filename) per model and quantisation.
# Repos chosen because they publish f16 alongside quantised variants, which is
# what makes the runtime-versus-quantisation split possible.
MODELS = {
    "Qwen2.5-1.5B": {
        "f16":    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-fp16.gguf"),
        "q8_0":   ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q8_0.gguf"),
        "q4_k_m": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    },
    "Qwen2.5-0.5B": {
        "f16":    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf"),
        "q8_0":   ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"),
        "q4_k_m": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    },
    # No fp16 GGUF is published for this repo, so the precision ladder is
    # incomplete here; kept for the quantised comparison only.
    "TinyLlama-1.1B": {
        "f16":    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.fp16.gguf"),
        "q8_0":   ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
        "q4_k_m": ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    },
}

CONTEXTS = [128, 512, 1024, 2048, 4096]
CANARY_CTX = 512
DECODE_TOKENS = 32
WARMUP = 2
REPS = 5
SEED = 20261001

FIELDS = [
    "model", "phase", "threads", "context_tokens", "reps",
    "median_ms", "iqr_ms", "min_ms", "max_ms", "ms_per_1k_ctx",
    "tokens_per_s", "quant", "backend", "elapsed_s", "timestamp",
]

STOP = False


def _sigint(signum, frame):
    global STOP
    if STOP:
        sys.exit("\nforced exit")
    STOP = True
    print("\n  stopping after this measurement; Ctrl-C again to force",
          flush=True)


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    try:
        ids, phys = set(), None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    ids.add((phys, line.split(":")[1].strip()))
        if ids:
            return len(ids)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def stats_of(xs):
    xs = sorted(xs)
    if len(xs) >= 4:
        q = statistics.quantiles(xs, n=4)
        iqr = q[2] - q[0]
    else:
        iqr = xs[-1] - xs[0]
    return statistics.median(xs), iqr, xs[0], xs[-1]


# --------------------------------------------------------------------------
# Timing. The low-level eval/sample API is used rather than create_completion
# because only it exposes the phase boundary.
# --------------------------------------------------------------------------

def time_prefill(llm, tokens):
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(tokens)
    return (time.perf_counter() - t0) * 1000.0


def time_decode(llm, tokens, n_tokens):
    """Seed the cache with the prompt, then time n_tokens single steps.

    The seeding eval is outside the timed region: decode cost means the
    incremental steps, not the prompt pass that precedes them.
    """
    llm.reset()
    llm.eval(tokens)
    tok = llm.sample()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        llm.eval([tok])
        tok = llm.sample()
    return ((time.perf_counter() - t0) * 1000.0) / n_tokens


def measure(llm, tokens, phase, warmup, reps, decode_tokens):
    if phase == "prefill":
        for _ in range(warmup):
            time_prefill(llm, tokens)
        return [time_prefill(llm, tokens) for _ in range(reps)]
    for _ in range(warmup):
        time_decode(llm, tokens, 4)
    return [time_decode(llm, tokens, decode_tokens) for _ in range(reps)]


# --------------------------------------------------------------------------

def completed(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") == "canary":
                continue
            done.add((row["model"], row["phase"], int(row["threads"]),
                      int(row["context_tokens"])))
    return done


def emit(writer, fh, name, phase, threads, ctx, samples, quant, elapsed):
    med, iqr, lo, hi = stats_of(samples)
    writer.writerow({
        "model": name, "phase": phase, "threads": threads,
        "context_tokens": ctx, "reps": len(samples),
        "median_ms": round(med, 3), "iqr_ms": round(iqr, 3),
        "min_ms": round(lo, 3), "max_ms": round(hi, 3),
        "ms_per_1k_ctx": round(med / ctx * 1000, 3) if phase != "decode" else "",
        "tokens_per_s": (round(ctx / (med / 1000), 2) if phase != "decode"
                         else round(1000 / med, 3)),
        "quant": quant, "backend": "llama.cpp",
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass
    return med


def run_one(base, quant, repo, fname, threads, contexts, writer, fh, args, done):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    name = f"{base}-{quant}"
    cells = [(c, p) for c in contexts for p in ("prefill", "decode")
             if (name, p, threads, c) not in done]
    if not cells:
        print(f"\n=== {name} === already complete")
        return

    print(f"\n=== {name} ===  {len(cells)} cells remaining", flush=True)
    try:
        path = hf_hub_download(repo_id=repo, filename=fname)
    except Exception as exc:
        print(f"    download failed: {str(exc)[:160]}")
        print(f"    check the filename on https://huggingface.co/{repo}")
        return

    maxctx = max(contexts) + args.decode_tokens + 64
    llm = Llama(model_path=path, n_ctx=maxctx, n_threads=threads,
                n_batch=2048, logits_all=False, verbose=False, seed=args.seed)
    vocab = llm.n_vocab()
    print(f"    vocab {vocab}, n_ctx {maxctx}, threads {threads}", flush=True)

    rng = random.Random(args.seed)
    prompts = {c: [rng.randrange(1, vocab) for _ in range(c)]
               for c in set(contexts) | {CANARY_CTX}}
    rng.shuffle(cells)

    since = 0
    for i, (ctx, phase) in enumerate(cells, 1):
        if STOP:
            break
        reps = args.reps if ctx <= 1024 else max(3, args.reps - 2)
        t0 = time.perf_counter()
        try:
            samples = measure(llm, prompts[ctx], phase, args.warmup, reps,
                              args.decode_tokens)
        except Exception as exc:
            print(f"    [{i}/{len(cells)}] {phase} ctx={ctx} FAILED: "
                  f"{str(exc)[:100]}", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        med = emit(writer, fh, name, phase, threads, ctx, samples, quant,
                   elapsed)
        rate = (ctx / (med / 1000)) if phase == "prefill" else (1000 / med)
        unit = "ms" if phase == "prefill" else "ms/tok"
        print(f"    [{i}/{len(cells)}] {phase:7s} ctx={ctx:5d}  "
              f"{med:9.2f} {unit:6s} ({rate:8.2f} tok/s)  [{elapsed:.0f}s]",
              flush=True)

        since += 1
        if since >= args.canary_every and not STOP:
            since = 0
            s = measure(llm, prompts[CANARY_CTX], "prefill", 1, 3,
                        args.decode_tokens)
            cmed = emit(writer, fh, name, "canary", threads, CANARY_CTX, s,
                        quant, 0)
            print(f"        canary prefill@{CANARY_CTX}: {cmed:.1f} ms",
                  flush=True)

    del llm


def compare(llama_csv, torch_csv):
    """Side-by-side phase gap: the result the paper needs."""
    if not os.path.exists(torch_csv):
        print(f"\n(no {torch_csv} alongside; skipping comparison)")
        return
    import collections

    def load(path, backend):
        out = collections.defaultdict(dict)
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["phase"] not in ("prefill", "decode"):
                    continue
                if backend == "torch" and int(r["threads"]) != 4:
                    continue
                try:
                    out[(r["model"], int(r["context_tokens"]))][r["phase"]] = \
                        float(r["tokens_per_s"])
                except (ValueError, KeyError):
                    pass
        return out

    print("\n" + "=" * 74)
    print("PHASE COST GAP BY RUNTIME  (decode tokens per prefill token)")
    for path, tag in ((torch_csv, "PyTorch fp32"), (llama_csv, "llama.cpp")):
        d = load(path, "torch" if tag.startswith("PyTorch") else "llama")
        for model in sorted({m for m, _ in d}):
            parts = []
            for ctx in sorted({c for m, c in d if m == model}):
                v = d[(model, ctx)]
                if "prefill" in v and "decode" in v and v["decode"]:
                    parts.append(f"{ctx}:{v['prefill']/v['decode']:.1f}x")
            if parts:
                print(f"  {tag:<14} {model.split('/')[-1]:<26} "
                      + "  ".join(parts))
    print("=" * 74)
    print("Read: if llama.cpp f16 holds the gap, the asymmetry is structural.")
    print("If it collapses, the PyTorch figure was largely framework cost.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--quants", nargs="*",
                    default=["f16", "q8_0", "q4_k_m"])
    ap.add_argument("--contexts", nargs="*", type=int, default=CONTEXTS)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    ap.add_argument("--canary-every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="llama_profile.csv")
    ap.add_argument("--compare-with", default="cpu_profile_merged.csv")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if args.compare_only:
        compare(args.out, args.compare_with)
        return

    try:
        import llama_cpp
    except ImportError:
        sys.exit("Missing llama-cpp-python. Install with:\n"
                 "  pip install llama-cpp-python huggingface_hub")

    try:
        signal.signal(signal.SIGINT, _sigint)
    except ValueError:
        pass

    # Threads must match the PyTorch run exactly, or the comparison is
    # between two configurations as well as two runtimes.
    threads = args.threads or (os.cpu_count() or 4)
    print(f"llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')}  "
          f"| {platform.platform()}")
    print(f"threads {threads} (physical cores {physical_cores()}), "
          f"contexts {args.contexts}")

    done = completed(args.out)
    if done:
        print(f"resuming: {len(done)} measurements already recorded")

    fresh = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for base in args.models:
            if base not in MODELS:
                print(f"unknown model {base}; known: {list(MODELS)}")
                continue
            for quant in args.quants:
                if STOP:
                    break
                if quant not in MODELS[base]:
                    continue
                repo, fname = MODELS[base][quant]
                try:
                    run_one(base, quant, repo, fname, threads, args.contexts,
                            writer, fh, args, done)
                except Exception as exc:
                    print(f"    [{base}-{quant}] FAILED: {str(exc)[:160]}")

    print(f"\nresults in {args.out}")
    compare(args.out, args.compare_with)
    if STOP:
        print("\nInterrupted. Rerun the same command to continue.")


if __name__ == "__main__":
    main()

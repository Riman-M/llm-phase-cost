#!/usr/bin/env python3
"""
cpu_profiler.py -- prefill/decode profiling of small LLMs on CPU.

Self-contained. Built to run on a laptop over several sittings and survive
being interrupted at any point.

    pip install torch transformers accelerate --index-url https://download.pytorch.org/whl/cpu
    python cpu_profiler.py

Resume after stopping it (Ctrl-C, reboot, closed lid) with the same command:
completed measurements are read back from the CSV and skipped.

------------------------------------------------------------------------------
WHAT IS MEASURED

  prefill : one forward pass over a prompt of C tokens, nothing generated.
            Compute-bound; parallel across the prompt.
  decode  : per-token latency of incremental generation with a KV cache warm
            from a C-token prompt. Memory-bandwidth-bound; serial.

They are timed apart because they scale differently with context length and
with thread count, and an end-to-end generation number hides exactly that.

Prompt CONTENT does not affect timing, only lengths do, so random
in-vocabulary token ids are used. This matches the released Azure inference
traces, which carry token counts and no prompt text for the same reason, and
means no corpus is needed.

------------------------------------------------------------------------------
MEASUREMENT HYGIENE

Three hazards on a personal machine, and what is done about each.

  Thermal / power throttling.  A laptop under sustained load slows down over
  minutes. Because context is swept in increasing order, drift would look
  like context scaling. Two defences: cell order is SHUFFLED within each
  model (fixed seed, so the run is reproducible), and a fixed reference cell
  is re-measured every --canary-every cells and written to the CSV with
  phase="canary". If those canary rows drift upward across the run, the
  session was throttling and should be rerun cooler. The summary prints the
  drift.

  Background load.  Close everything. A browser alone moves these numbers.
  Median of several repetitions is reported rather than mean, so one
  interrupted pass does not drag the estimate.

  Cold start.  The first pass through any cell pays lazy allocation and cache
  warming and can run several times the steady state, so --warmup passes are
  discarded before every measured cell.

------------------------------------------------------------------------------
OUTPUT

  cpu_profile.csv            one row per measured cell, appended as it goes
  cpu_profile_machine.json   host description, for the paper's setup section

Re-running with --summary-only reprints the analysis without measuring.
"""

import argparse
import csv
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

MODELS = [
    ("HuggingFaceTB/SmolLM2-360M", 0.36),
    ("Qwen/Qwen2.5-0.5B", 0.49),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1.10),
    ("Qwen/Qwen2.5-1.5B", 1.54),
]
CONTEXTS = [128, 256, 512, 1024, 2048, 4096]
CANARY_CTX = 512                 # reference cell for drift detection
DECODE_TOKENS = 32
WARMUP = 2
REPS = 5
SEED = 20261015

FIELDS = [
    "model", "phase", "threads", "context_tokens", "reps",
    "median_ms", "iqr_ms", "min_ms", "max_ms", "ms_per_1k_ctx",
    "tokens_per_s", "params_millions", "elapsed_s", "timestamp",
]

STOP = False


def _sigint(signum, frame):
    """First Ctrl-C finishes the current cell and exits cleanly; second kills."""
    global STOP
    if STOP:
        sys.exit("\nforced exit")
    STOP = True
    print("\n  stopping after this measurement; press Ctrl-C again to force",
          flush=True)


# --------------------------------------------------------------------------
# Host description
# --------------------------------------------------------------------------

def physical_cores():
    """Physical core count, which is what prefill parallelism actually scales on.

    Logical count includes hyperthreads and would make speedup look bad.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            ids = set()
            phys = None
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
    elif sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"]).strip())
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores"],
                stderr=subprocess.DEVNULL).decode()
            nums = [int(x) for x in out.split() if x.isdigit()]
            if nums:
                return sum(nums)
        except Exception:
            pass
    return max(1, (os.cpu_count() or 2) // 2)


def available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def cpu_name():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass
    elif sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    return platform.processor() or "unknown"


def machine_info():
    import torch
    ram = available_ram_gb()
    return {
        "cpu_model": cpu_name(),
        "physical_cores": physical_cores(),
        "logical_cpus": os.cpu_count(),
        "available_ram_gb": round(ram, 1) if ram else None,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def stats_of(xs):
    xs = sorted(xs)
    if len(xs) >= 4:
        q = statistics.quantiles(xs, n=4)
        iqr = q[2] - q[0]
    else:
        iqr = xs[-1] - xs[0]
    return statistics.median(xs), iqr, xs[0], xs[-1]


def time_prefill(model, ids):
    import torch
    with torch.inference_mode():
        t0 = time.perf_counter()
        model(ids)
        return (time.perf_counter() - t0) * 1000.0


def time_decode(model, ids, n_tokens):
    """Seed the KV cache with the prompt, then time n_tokens single steps.

    The seeding pass is deliberately excluded: decode cost means the
    incremental steps, not the prompt pass that precedes them.
    """
    import torch
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed / n_tokens


def measure(model, ids, phase, warmup, reps, decode_tokens):
    if phase == "prefill":
        for _ in range(warmup):
            time_prefill(model, ids)
        return [time_prefill(model, ids) for _ in range(reps)]
    for _ in range(warmup):
        time_decode(model, ids, 4)
    return [time_decode(model, ids, decode_tokens) for _ in range(reps)]


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------

def completed(path):
    done = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("phase") == "canary":
                    continue
                done.add((row["model"], row["phase"], int(row["threads"]),
                          int(row["context_tokens"])))
    except Exception as exc:
        print(f"warning: could not read existing results ({exc})")
    return done


def emit(writer, fh, name, phase, threads, ctx, samples, params_m, elapsed):
    med, iqr, lo, hi = stats_of(samples)
    writer.writerow({
        "model": name, "phase": phase, "threads": threads,
        "context_tokens": ctx, "reps": len(samples),
        "median_ms": round(med, 3), "iqr_ms": round(iqr, 3),
        "min_ms": round(lo, 3), "max_ms": round(hi, 3),
        "ms_per_1k_ctx": round(med / ctx * 1000, 3) if phase != "decode" else "",
        "tokens_per_s": (round(ctx / (med / 1000), 2) if phase != "decode"
                         else round(1000 / med, 3)),
        "params_millions": round(params_m, 1),
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    fh.flush()
    try:
        os.fsync(fh.fileno())      # survive a hard power loss, not just Ctrl-C
    except OSError:
        pass
    return med


# --------------------------------------------------------------------------
# Per-model run
# --------------------------------------------------------------------------

def run_model(name, params_b, threads_list, contexts, writer, fh, args, done):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    need = params_b * 4 * 1.6          # fp32 weights plus activation headroom
    ram = available_ram_gb()
    if ram is not None and need > ram and not args.allow_big:
        print(f"\n=== {name} === SKIPPED: needs about {need:.1f} GB, "
              f"{ram:.1f} GB available. Close applications, or pass "
              f"--allow-big to try anyway.")
        return

    cells = [(t, c, p) for t in threads_list for c in contexts
             for p in ("prefill", "decode")
             if (name, p, t, c) not in done]
    if not cells:
        print(f"\n=== {name} === already complete")
        return

    print(f"\n=== {name} ===  {len(cells)} cells remaining", flush=True)
    cfg = AutoConfig.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32, low_cpu_mem_usage=True)
    except TypeError:                   # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    vocab = cfg.vocab_size
    print(f"    {params_m:.0f}M parameters, vocab {vocab}", flush=True)

    # Shuffle so any slow drift over the session does not line up with
    # context length and masquerade as context scaling.
    rng = random.Random(args.seed)
    rng.shuffle(cells)

    g = torch.Generator().manual_seed(args.seed)
    prompts = {c: torch.randint(0, vocab, (1, c), generator=g)
               for c in set(contexts) | {CANARY_CTX}}

    since_canary = 0
    for i, (threads, ctx, phase) in enumerate(cells, 1):
        if STOP:
            break
        torch.set_num_threads(threads)
        reps = args.reps if ctx <= 1024 else max(3, args.reps - 2)

        t0 = time.perf_counter()
        try:
            samples = measure(model, prompts[ctx], phase, args.warmup, reps,
                              args.decode_tokens)
        except (RuntimeError, MemoryError) as exc:
            print(f"    [{i}/{len(cells)}] {phase} t={threads} ctx={ctx} "
                  f"FAILED: {str(exc)[:80]}", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        med = emit(writer, fh, name, phase, threads, ctx, samples, params_m,
                   elapsed)

        unit = "ms" if phase == "prefill" else "ms/tok"
        rate = (ctx / (med / 1000)) if phase == "prefill" else (1000 / med)
        print(f"    [{i}/{len(cells)}] {phase:7s} t={threads} ctx={ctx:5d}  "
              f"{med:9.1f} {unit:6s} ({rate:7.2f} tok/s)  "
              f"[{elapsed:.0f}s]", flush=True)

        since_canary += 1
        if since_canary >= args.canary_every and not STOP:
            since_canary = 0
            s = measure(model, prompts[CANARY_CTX], "prefill", 1, 3,
                        args.decode_tokens)
            cmed = emit(writer, fh, name, "canary", threads, CANARY_CTX, s,
                        params_m, 0)
            print(f"        canary prefill@{CANARY_CTX}: {cmed:.1f} ms",
                  flush=True)

        if args.cooldown:
            time.sleep(args.cooldown)

    del model


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _num(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return None


def summarise(path):
    if not os.path.exists(path):
        print("no results yet")
        return
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("no results yet")
        return

    canary = [r for r in rows if r["phase"] == "canary"]
    data = [r for r in rows if r["phase"] in ("prefill", "decode")]
    models = sorted({r["model"] for r in data})

    def pick(m, ph, t, c):
        v = [_num(r, "tokens_per_s") for r in data
             if r["model"] == m and r["phase"] == ph
             and int(r["threads"]) == t and int(r["context_tokens"]) == c]
        return v[0] if v else None

    print("\n" + "=" * 74)
    print("THROUGHPUT (tokens/s)")
    for m in models:
        threads = sorted({int(r["threads"]) for r in data if r["model"] == m})
        ctxs = sorted({int(r["context_tokens"]) for r in data
                       if r["model"] == m})
        print(f"\n  {m}")
        print("    ctx    " + "".join(f"{'pre t'+str(t):>11s}"
                                      f"{'dec t'+str(t):>11s}"
                                      for t in threads))
        for c in ctxs:
            line = f"    {c:<7d}"
            for t in threads:
                for ph in ("prefill", "decode"):
                    v = pick(m, ph, t, c)
                    line += f"{v:>11.2f}" if v else f"{'-':>11s}"
            print(line)

    print("\n" + "=" * 74)
    print("PHASE COST GAP  (decode tokens per prefill token)")
    for m in models:
        ctxs = sorted({int(r["context_tokens"]) for r in data
                       if r["model"] == m})
        for t in sorted({int(r["threads"]) for r in data if r["model"] == m}):
            parts = []
            for c in ctxs:
                p, d = pick(m, "prefill", t, c), pick(m, "decode", t, c)
                if p and d:
                    parts.append(f"{c}:{p/d:.1f}x")
            if parts:
                print(f"  {m.split('/')[-1]:<30s} t={t}  " + "  ".join(parts))

    print("\n" + "=" * 74)
    print("PARALLEL SPEEDUP  (highest thread count over 1)")
    for m in models:
        ts = sorted({int(r["threads"]) for r in data if r["model"] == m})
        if len(ts) < 2:
            continue
        hi, lo = ts[-1], ts[0]
        ctxs = sorted({int(r["context_tokens"]) for r in data
                       if r["model"] == m})
        for ph in ("prefill", "decode"):
            parts = []
            for c in ctxs:
                a, b = pick(m, ph, hi, c), pick(m, ph, lo, c)
                if a and b:
                    parts.append(f"{c}:{a/b:.2f}x")
            if parts:
                print(f"  {m.split('/')[-1]:<30s} {ph:8s} t{hi}/t{lo}  "
                      + "  ".join(parts))

    print("\n" + "=" * 74)
    if canary:
        print("THERMAL DRIFT CHECK  (reference cell re-measured during the run)")
        # Canary rows inherit the thread count of the cell they follow, so
        # they MUST be compared within a thread setting. Pooling them across
        # thread counts measures the parallel speedup and reports it as drift.
        for m in sorted({r["model"] for r in canary}):
            for t in sorted({int(r["threads"]) for r in canary
                             if r["model"] == m}):
                vals = [v for v in (_num(r, "median_ms") for r in canary
                                    if r["model"] == m
                                    and int(r["threads"]) == t) if v]
                if len(vals) < 2:
                    print(f"  {m.split('/')[-1]:<30s} t={t}  only "
                          f"{len(vals)} sample(s), need 2 to judge drift")
                    continue
                drift = (vals[-1] - vals[0]) / vals[0] * 100
                verdict = ("stable" if abs(drift) < 5
                           else "SUSPECT -- machine was not thermally steady")
                print(f"  {m.split('/')[-1]:<30s} t={t}  "
                      f"first {vals[0]:8.1f} ms   last {vals[-1]:8.1f} ms   "
                      f"drift {drift:+6.1f}%   {verdict}")
    else:
        print("THERMAL DRIFT CHECK: no canary rows yet (needs a longer run)")
    print("=" * 74)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Profile prefill and decode cost of small LLMs on CPU.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="HuggingFace ids; default is the built-in ladder")
    ap.add_argument("--contexts", nargs="*", type=int, default=CONTEXTS)
    ap.add_argument("--threads", nargs="*", type=int, default=None,
                    help="thread counts; default physical cores then 1")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    ap.add_argument("--canary-every", type=int, default=6,
                    help="re-measure the reference cell every N cells")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to idle between cells; try 10 on a laptop")
    ap.add_argument("--allow-big", action="store_true",
                    help="attempt models larger than free RAM suggests")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="cpu_profile.csv")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        summarise(args.out)
        return

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("Missing dependencies. Install with:\n"
                 "  pip install torch transformers accelerate "
                 "--index-url https://download.pytorch.org/whl/cpu")

    try:
        signal.signal(signal.SIGINT, _sigint)
    except ValueError:
        pass

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)

    info = machine_info()
    with open(os.path.splitext(args.out)[0] + "_machine.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))

    cores = info["physical_cores"]
    # Highest thread count first: if the run is cut short, the
    # deployment-realistic configuration is the one already collected.
    threads_list = args.threads or sorted({cores, 1}, reverse=True)
    print(f"\nthread sweep: {threads_list}   contexts: {args.contexts}")

    models = ([(m, 1.0) for m in args.models] if args.models else MODELS)
    done = completed(args.out)
    if done:
        print(f"resuming: {len(done)} measurements already recorded")

    fresh = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for name, params_b in models:
            if STOP:
                break
            try:
                run_model(name, params_b, threads_list, args.contexts,
                          writer, fh, args, done)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"    [{name}] FAILED: {exc}", flush=True)

    print(f"\nresults in {args.out}")
    summarise(args.out)
    if STOP:
        print("\nInterrupted. Rerun the same command to continue where this "
              "left off.")


if __name__ == "__main__":
    main()

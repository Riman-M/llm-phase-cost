# Phase cost structure of LLM inference, and the workload dependence of quantisation

Measurement scripts and raw results for *Diminishing Returns: Why
Prefill-Dominated Workloads Erode the Benefit of Low-Precision LLM Inference*.

The study has two independent halves that meet only at the end. Four Azure
production inference traces supply per-request token counts; a set of open
models profiled under three weight precisions supplies the cost of each
inference phase as a function of context length. Combining them gives the
share of service time each phase occupies in real traffic, and what a given
workload actually gains from quantisation.

## Summary of findings

Decode cost is set by the bytes of model weights read per generated token, and
that read proceeds at memory bandwidth. PyTorch at float32 and llama.cpp at
16-bit weights differ by 2.03x in decode latency but recover 18.4 and
18.6 GB/s respectively once byte counts are divided out. Prefill is
compute-bound and gains little from reduced precision.

Consequently the speedup from 4-bit quantisation depends on the workload, not
on the technique:

| Workload | Prefill share (fp32) | Q4\_K\_M speedup | f16 vs fp32 |
| --- | --- | --- | --- |
| 2023 conversation | 32.4 % | 2.13x | 1.16x |
| 2024 conversation | 55.6 % | 1.49x | 0.88x |
| 2023 code | 86.2 % | 1.06x | 0.66x |
| 2024 code | 90.2 % | 1.01x | 0.64x |

The final column is a negative result: moving to a 16-bit runtime makes the
prefill-dominated code workloads 34–36 % *slower*, because it surrenders the
vendor BLAS kernels that dominate prefill.

## Layout

```
scripts/     measurement and analysis code
results/     raw outputs the paper's tables and figures are computed from
figures/     the four figures, as vector PDFs
```

`results/superseded/` holds earlier measurement sweeps that the paper does not
draw numbers from. They are kept because the cross-session reproducibility
finding in Section 5 depends on them.

## Reproducing

### 1. Traces

```
python scripts/trace_prep.py --download --outdir results
```

Downloads the four traces from the Azure public dataset (CC-BY-4.0) and writes
characterisation statistics and arrival series. The 2024 files are about 1 GB
each and are streamed in chunks, so peak memory stays near 1 GB regardless of
input size. Budget 2 GB of disk.

```
python scripts/ratio_stability.py --datadir data --outdir results
```

Tests whether the prefill:decode ratio is stable across the hours of the 2024
week, which is what defeats the objection that the one-hour 2023 samples are
unrepresentative.

### 2. Phase profiling

**Both runtimes must be profiled in a single session on one machine.** This is
not a convenience. Measurements taken in two sessions on nominally identical
cloud hosts disagreed by up to 39 %, with the discrepancy differing by phase
and by quantisation level, so no single calibration factor could reconcile
them. Compute-bound prefill is sensitive to co-tenancy on execution units;
bandwidth-bound decode is not. The scripts emit periodic re-measurements of a
fixed reference cell (`phase=canary` rows) so drift is visible rather than
silent.

```
pip install -r requirements.txt

python scripts/cpu_profiler.py   --models Qwen/Qwen2.5-1.5B \
    --contexts 128 512 1024 2048 4096 8192 --threads 4 --reps 3 \
    --out joint_torch.csv

python scripts/llama_profiler.py --models Qwen2.5-1.5B --quants f16 q4_k_m \
    --contexts 128 512 1024 2048 4096 8192 --threads 4 --reps 3 \
    --out joint_llama.csv
```

Roughly five hours on two physical cores. Both scripts checkpoint per cell and
resume from the same command if interrupted. `--summary-only` reprints the
analysis without measuring.

For the four-model ladder in Table 2, drop `--models` to sweep the default set.

### 3. Cost model

```
python scripts/cost_model.py --profile joint_torch.csv --traces data/*.csv \
    --model Qwen/Qwen2.5-1.5B      --threads 4 --outdir results_fp32
python scripts/cost_model.py --profile joint_llama.csv --traces data/*.csv \
    --model Qwen2.5-1.5B-f16       --threads 4 --outdir results_f16
python scripts/cost_model.py --profile joint_llama.csv --traces data/*.csv \
    --model Qwen2.5-1.5B-q4_k_m    --threads 4 --outdir results_q4
```

A separate `--outdir` per run, since the output filename is fixed. Windows
users can run `scripts/run_cost_models.bat`, which checks inputs first and logs
all three runs together.

### 4. Figures

```
python scripts/make_figures.py
```

Values are transcribed from the measurement outputs rather than recomputed, so
the figures cannot drift from the tables without the transcription being
visibly wrong. Each figure names its source file in a comment.

## Measurement notes

Prefill is timed as one forward pass over a prompt of C tokens with nothing
generated. Decode is timed as the per-token latency of single-token steps with
a KV cache warm from a C-token prompt; the seeding pass is excluded. Prompt
*content* does not affect timing, only lengths do, so prompts are random
in-vocabulary token ids and no corpus is needed. This matches the Azure traces,
which carry token counts and no prompt text for the same reason.

Each cell discards two warmup passes and reports a median; cell order is
shuffled within each configuration under a fixed seed, so that drift over a
session cannot align with context length and be mistaken for context scaling.

## Environment

Intel Xeon @ 2.20 GHz, 2 physical cores / 4 logical CPUs, 32 GB RAM, Linux
6.18, Python 3.12.13, PyTorch 2.10.0 (CPU build), llama.cpp via
llama-cpp-python. All measurement ran on a free-tier cloud CPU notebook.

Models: SmolLM2-360M, Qwen2.5-0.5B, TinyLlama-1.1B-Chat-v1.0, Qwen2.5-1.5B, in
safetensors and GGUF form, retrieved at pinned revisions.

## Limitations

These are stated in full in the paper. The two that most affect reuse of this
code: all measurement is on CPU, where the bandwidth-to-compute ratio differs
from GPU, so the mechanism transfers but the specific crossover does not; and
capacity is modelled as a single replica serving one request at a time, whereas
production engines overlap the phases through continuous batching.

## Licence

Code under MIT (`LICENSE`). The Azure inference traces are CC-BY-4.0 and are
not redistributed here; `trace_prep.py` fetches them from the original source.
Model weights remain under their respective licences.

## Citation

See `CITATION.cff`.

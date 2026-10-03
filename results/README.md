# Raw measurement outputs

## Primary set

All numbers in the paper come from these files. `joint_torch.csv` and
`joint_llama.csv` were measured in one session on one host, which is required
for the runtime comparison to be valid (see the note on cross-session drift in
the top-level README).

| File | Contents |
| --- | --- |
| `joint_torch.csv` | PyTorch float32 phase grid, Qwen2.5-1.5B, contexts 128–8192 |
| `joint_llama.csv` | llama.cpp f16 and Q4_K_M phase grids, same model and contexts |
| `joint_torch_machine.json` | Host description for that session |
| `trace_stats.csv` | Workload characterisation of the four Azure traces |
| `cost_model_fp32.csv` | Cost model output under PyTorch float32 |
| `cost_model_f16.csv` | Cost model output under llama.cpp f16 |
| `cost_model_q4.csv` | Cost model output under llama.cpp Q4_K_M |
| `cost_models_log.txt` | Combined console log of all three cost model runs |

Rows with `phase=canary` are periodic re-measurements of a fixed reference
cell, used to detect drift within a session. They must be compared within a
thread count, since they inherit the thread setting of the cell preceding them.

## superseded/

Earlier sweeps, retained because the cross-session reproducibility finding
depends on them. The paper draws no numbers from these files.

| File | Why it was superseded |
| --- | --- |
| `cpu_profile_merged.csv` | Four-model PyTorch ladder plus a calibrated 8192 cell from a second session |
| `cpu_profile_ext.csv` | Second-session extension whose 4096 overlap disagreed with the first by up to 39 % |
| `llama_profile.csv` | First llama.cpp sweep, measured in a different session from the PyTorch baseline |

The four-model ladder in `cpu_profile_merged.csv` is the source of Table 2,
which compares models within one runtime and is therefore unaffected by the
cross-session problem.

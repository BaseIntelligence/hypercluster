# Scoring and raw weights

Hypercluster uses a **fixed** four-factor product. Do not invent alternate product formulas.

## Formula (per attempt)

```text
composite = correctness × efficiency × fabric_gate × tee_bonus
```

| Factor | Type | Definition |
| --- | --- | --- |
| `correctness` | gate `{0,1}` | Golden microbench / output digest / schema validation. Zero zeroes all |
| `efficiency` | continuous ≥ 0 | Compute-normalized metric. Wall clock is a **cap**, not the primary score |
| `fabric_gate` | gate `{0,1}` | One when required fabric is present without forbidden fallback; else zero |
| `tee_bonus` | multiplier ≥ 1 | `1.0` without verified TEE; higher only when proof verifies under policy |

### Integrity zero

Any integrity failure forces `composite = 0` for that attempt, including attestation fail, image mutation, rank desync, inventory spoof, fabric lie, and **GPU probe honesty fails** (failed/error evidence under live-evidence policy, or claim-vs-measured mismatch vs last good `GpuHostEvidence`).

GPU probe **does not add a fifth formula factor**. Defaults leave the unprobed sim path green; operators may inject zeros with `HYPER_SIM_GPU_PROBE_FAIL` for CI drills. See [GPU probe](gpu-probe.md).

## Aggregation → raw weights

1. Select a scoring window (for example last N attempts via `HYPER_SCORE_WINDOW_ATTEMPTS`).
2. Aggregate demand and/or supply composites per hotkey (default: sum of positive composites).
3. Apply soft penalties (self-deal rate, spam fails) as policy.
4. Emit `dict[hotkey, float]` with **finite values ≥ 0**.
5. Prefer burn-safe empty epochs over inventing fake mass.
6. Challenge **never** calls `set_weights`; Base master normalizes emission and maps hotkeys → UIDs.

## Weight push

Background or CLI-triggered push builds a monochronic snapshot:

- `epoch`, increasing `revision`
- tz-aware `computed_at` / `expires_at`
- `payload_digest` over canonical payload
- POST raw weights to master (local mock supported for development)

Serve the same map via Base `get_weights_fn` / internal GET.

## Suggested knobs

```text
HYPER_TEE_BONUS_TDX=1.08
HYPER_TEE_BONUS_TDX_GPU=1.20
HYPER_WEIGHT_PUSH_INTERVAL_S=120
HYPER_SCORE_WINDOW_ATTEMPTS=50
HYPER_EFFICIENCY_FLOOR=0.0
HYPER_MASTER_BASE_URL=http://127.0.0.1:3201
```

## HTTP routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/leaderboard` | Aggregated composites |
| GET | `/v1/scores/{hotkey}` | History |
| GET | `/v1/weight-preview` | Pending/latest raw map (may be restricted in deploy) |

## CLI

```bash
hypercluster score show --hotkey HOTKEY
hypercluster score recompute [--epoch N]
hypercluster weights preview
hypercluster weights push --epoch E --revision R
```

Never print challenge tokens in CLI output.

## Local validation

```bash
hypercluster sim run-scenario --name weights --url http://127.0.0.1:3200
# with mock master env if needed:
# HYPER_MASTER_BASE_URL=http://127.0.0.1:3201
```

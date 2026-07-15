# TEE offline verification

Hypercluster includes a dstack-oriented **offline** attestation path mandatory for CI. Live silicon verification is optional and finish-closed when hardware is absent.

## Modes

| Mode | Behavior |
| --- | --- |
| `offline_fixture` | Golden quotes, compose hashes, mutated negatives (CI) |
| `sim` | Sim-tier proofs; must never claim **live** tee_bonus |
| `live` | Optional remote verifier when `HYPER_TEE_LIVE=1` and hardware/credits exist |

## Pipeline (high level)

```text
1. Bind report_data layout (job id ‖ image digest ‖ nonce)
2. Select backend: offline fixture | sim | live
3. Check compose_hash / measurement allowlist
4. Apply TCB / advisory policy (fail closed when enforce is on)
5. Optional GPU evidence nonce echo for tdx+gpu_cc
6. Persist verdict on job_proofs; tee_bonus only when valid
```

## Bonus application

`tee_bonus` is a multiplier **≥ 1.0**:

| Claim | Typical multiplier (config-pinned) |
| --- | --- |
| none / unverified | `1.0` (no boost; cheat paths may zero composite) |
| verified `tdx` | e.g. `HYPER_TEE_BONUS_TDX=1.08` |
| verified `tdx+gpu_cc` | e.g. `HYPER_TEE_BONUS_TDX_GPU=1.20` |

Unverified TEE **claims** never receive inflated live bonus. See [Scoring](scoring.md).

## Residual risk language

- Offline green proves verifier wiring and golden fixture stability.
- Live path is optional and may be unavailable on many hosts.
- Multi-node IB is **not** encrypted by TDX/GPU CC alone. Prefer "measured collocated compute" rather than claiming encrypted multi-node fabrics.

## CLI

```bash
hypercluster attest verify-offline --quote-fixture tests/fixtures/tee/...
hypercluster attest compose-hash --compose-file tests/fixtures/tee/golden_compose.yml \
  --check-golden tests/fixtures/tee/golden_compose.sha256
# verify-live is skip-safe without hardware
```

## Local validation

```bash
hypercluster sim run-scenario --name tee-offline --url http://127.0.0.1:3200
uv run pytest -q tests/attest
```

Fixtures live under `tests/fixtures/tee/`.

# Jobs

HyperJobs are Modal-like multi-node work units admitted into a SQLite-backed queue with topology-aware placement and post-run scoring.

## Lifecycle

```text
submitted
  → admitted          # static gates: budgets, allowlist, nnodes sanity
  → placing           # planner rankmap + NCCL env
  → provisioning      # wait pod / inventory healthy
  → running           # launcher fan-out
  → collecting        # metrics, fabric digests, optional proofs
  → scoring           # four-factor composite
  → succeeded | failed | timeout | cancelled
       └── teardown lease resources per policy
```

## Spec shape (conceptual)

| Field | Notes |
| --- | --- |
| `image_digest` | Content-addressed / pinned image |
| `entrypoint` | Command vector |
| `world_size`, `nnodes`, `nproc_per_node` | Distributed layout |
| `backend` | `nccl` (default) or `gloo` |
| `fabric` | `auto` \| `ib` \| `eth` \| `nvlink_only` |
| `tee` | `none` \| `tdx` \| `tdx+gpu_cc` |
| `placement_policy` | `pack` \| `spread` |
| `timeout_s` | Hard deadline (wall clock is a cap, not the primary score) |
| `resource` | Capacity / price / lifetime bounds |

## HTTP routes

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/jobs` | Submit HyperJob (signed) |
| GET | `/v1/jobs` | List submitter jobs (identity-scoped list) |
| GET | `/v1/jobs/{job_id}` | Status, placement, summary |
| POST | `/v1/jobs/{job_id}/cancel` | Cancel if non-terminal (signed) |
| GET | `/v1/jobs/{job_id}/attempts/{n}` | Attempt metrics digests |
| GET | `/v1/jobs/{job_id}/fabric-report` | Job fabric view |
| POST | `/v1/jobs/{job_id}/results` | Provider/worker result envelope + proofs |

## Queue and scaling notes

| Concern | Approach |
| --- | --- |
| Queue claim | Status CAS / atomic lease style on SQLite |
| Concurrency | Cap concurrent multi-node worlds; multiplex small jobs |
| Combined worker | `HYPER_COMBINED_WORKER=true` drains place/launch/score loops in the API process |
| Idempotency | Client request ids on create; result posts keyed by attempt |
| Warm capacity | Provider-side healthy nodes for fast bind (not master-side container spawn) |

## CLI

```bash
hypercluster jobs submit --spec path/to/job.json
hypercluster jobs status --id JOB
hypercluster jobs list
hypercluster jobs cancel --id JOB
hypercluster jobs logs --id JOB [--attempt N]
```

Connection or API-down cases exit non-zero. Logs command prints safe digests/URI stubs and never floods secrets.

## Local validation

```bash
hypercluster sim run-scenario --name smoke --url http://127.0.0.1:3200
hypercluster sim run-scenario --name nccl --url http://127.0.0.1:3200
# Extended cross-area path (when API is up):
hypercluster sim run-scenario --name cross-happy-path --url http://127.0.0.1:3200
```

See [Architecture](architecture.md), [Fabric](fabric.md), and [Scoring](scoring.md).

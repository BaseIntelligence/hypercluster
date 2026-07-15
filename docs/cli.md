# CLI reference

Entry point: `hypercluster` (package script → `hypercluster.cli:app`).

```bash
uv sync --all-extras
uv run hypercluster --help
# After [project.scripts] changes: uv sync so .venv/bin/hypercluster is current
```

Shared flags typically include `--base-url` / `--url`, hotkey, and auth material paths. Never print tokens.

## Top-level commands

| Command | Purpose |
| --- | --- |
| `serve` | Dev uvicorn bind (default local host port configurable; containers use 8000) |
| `version` | Package version; with `--url`, match live `/version` |
| `health` | Probe live `/health`; exit 0 only when `status=ok` |
| `db init` / `db migrate` | Ensure SQLite schema under `CHALLENGE_DATABASE_URL` |
| `marketplace …` | Offers, rent, leases |
| `nodes …` | Register, heartbeat, fabric-scan, GPU probe + evidence |
| `jobs …` | Submit, status, list, cancel, logs |
| `fabric …` | Plan dry-run, gated launch, report show |
| `attest …` | Offline verify, compose-hash |
| `score …` | Recompute, show hotkey |
| `weights …` | Preview, push (never set_weights) |
| `sim …` | Doctor, seed, run-scenario |

## Marketplace

```text
hypercluster marketplace offers list
hypercluster marketplace offer create ...
hypercluster marketplace rent --offer-id ...
hypercluster marketplace lease show --id ...
hypercluster marketplace terminate --lease-id ...
```

## Nodes

```text
hypercluster nodes register ...
hypercluster nodes heartbeat ...
hypercluster nodes fabric-scan --node-id ...
hypercluster nodes probe-gpu NODE_ID [--mode full|quick] [--json]
hypercluster nodes probe-gpu-sim NODE_ID --pass-all|--fail CHECK_ID [--json]
hypercluster nodes evidence list|latest NODE_ID [--json]
hypercluster nodes evidence show EVIDENCE_ID [--json]
```

GPU probe CLI wraps the same product APIs. Default CI exercises **FakeSsh** fixtures; live RealSSH needs operator key paths (never PEM on the CLI argv for persistence). Exit codes for probe-gpu: **0** pass, **2** failed checks, **3** transport/error. See [GPU probe](gpu-probe.md).

## Jobs

```text
hypercluster jobs submit --spec job.json
hypercluster jobs status --id ...
hypercluster jobs list
hypercluster jobs cancel --id ...
hypercluster jobs logs --id ...
```

## Fabric / attest / score / weights

```text
hypercluster fabric plan --spec plan.json
hypercluster fabric report show --job-id ...
hypercluster attest verify-offline ...
hypercluster attest compose-hash --compose-file ... --check-golden ...
hypercluster score show --hotkey ...
hypercluster score recompute
hypercluster weights preview
hypercluster weights push --epoch ... --revision ...
```

## Simulator

```text
hypercluster sim doctor [--offline] [--url URL]
hypercluster sim seed --seed 0
hypercluster sim run-scenario --name smoke|marketplace|nccl|tee-offline|weights [--url URL]
```

Canonical suite order is fixed:

1. `smoke`
2. `marketplace`
3. `nccl`
4. `tee-offline`
5. `weights`

Extended cross scenarios (happy path, multinode fabric+TEE, market resilience, worker durability, weights/leaderboard chaos, docker proxy) use the same `run-scenario` entry with different names; they are local/sim helpers, not default cloud spend.

## Exit codes and safety

- Non-zero on API connection failure for status/list when the server is down.
- Auth reject paths must surface clearly (no silent private admin dumps).
- CLI commands must not emit secrets/tokens in normal or error printouts.

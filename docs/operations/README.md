# Operations

## Supported run paths

### Local API (dev)

```bash
uv sync --all-extras
export CHALLENGE_SHARED_TOKEN=dev-token
export CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////tmp/hypercluster-dev/challenge.sqlite3
mkdir -p /tmp/hypercluster-dev
uv run hypercluster serve --host 127.0.0.1 --port 3200
```

Identity checks:

```bash
curl -fsS http://127.0.0.1:3200/health
curl -fsS http://127.0.0.1:3200/ready
curl -fsS http://127.0.0.1:3200/version
```

Combined worker (API process also drains place/launch/score loops):

```bash
export HYPER_COMBINED_WORKER=true
export HYPER_COMBINED_WORKER_INTERVAL_SECONDS=1
```

### Mock master (weight push integration)

Run a lightweight mock about port 3201 in development, then:

```bash
export HYPER_MASTER_BASE_URL=http://127.0.0.1:3201
hypercluster weights push --epoch 1 --revision 1 --url http://127.0.0.1:3200
```

### Docker

```bash
# Stage Base wheel offline if needed, then:
docker compose up --build
curl -fsS http://127.0.0.1:3250/health
```

Container listens **8000** internally; compose maps host **3250**. SQLite /data is volume-backed.

## Configuration

See `.env.example` for `CHALLENGE_*` and `HYPER_*` knobs. Prefer secret files for tokens. Do not place third-party broker credentials in the challenge process env.

## Quality gate

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
```

Docker lifecycle tests that bind fixed host ports should run serially (`-n0`) when using xdist.

## Sim doctor

```bash
hypercluster sim doctor --offline
hypercluster sim doctor --url http://127.0.0.1:3200
```

## Clean shutdown

- Stop API processes cleanly (`Ctrl-C` or port-bound stop scripts you own).
- Remove throwaway docker containers/volumes after local experiments.
- For any live external rental used during maintainer QA: **always discontinue** and terminate leases. See [Live QA](../live-qa.md).

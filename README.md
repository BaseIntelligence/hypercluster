# Hypercluster

Compute power challenge for [Base Intelligence](https://github.com/BaseIntelligence): miners both **supply** cluster capacity and **demand** multi-node jobs through a **home-grown** marketplace (not a commercial cloud adapter).

## Who this is for

| Role | Path |
| --- | --- |
| **Miners (demand)** | Browse offers, rent capacity, submit Modal-like jobs via signed API/CLI |
| **Providers (supply)** | Register nodes (self-owned SSH fleets / your own inventory), list offers, host pods |
| **Operators / CI** | Local sim + SQLite; default tests never call commercial clouds |

## Miner setup (no commercial cloud account required)

1. Run or join a Hypercluster challenge API (`hypercluster serve` / Docker).
2. Register a hotkey-signed provider or act as a demand miner.
3. Offer capacity from **your** nodes, or rent listed marketplace capacity.
4. Submit jobs; scores flow to Base raw weights.

**You do not need a Verda (or any other commercial GPU broker) account to mine.** Cloud rentals are optional ops infrastructure a provider may choose off-band; Hypercluster only sees the node through the home-grown marketplace APIs.

## Quick start

```bash
uv sync --all-extras
export CHALLENGE_SHARED_TOKEN=dev-token
uv run hypercluster sim doctor --offline
uv run pytest -q
```

Identity surfaces (no miner auth): `GET /health`, `GET /ready`, `GET /version`.

## Boundaries (product)

- No Verda SDK, OAuth client, or `api.verda.com` adapter ships in this package.
- Default CI and `pytest` use local sim only (no live cloud purchases).
- Multi-node InfiniBand / NCCL is validated via the local simulator.
- Optional live single-GPU ops QA (maintainers only) lives outside the product tree.

## License

Apache-2.0 — see `LICENSE`.

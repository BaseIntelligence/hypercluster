# Miner guide (demand)

Demand miners browse Hypercluster marketplace offers, rent capacity, and submit multi-node (or single-node) jobs. Scores roll into raw weights for Base master emission share.

**You do not need a commercial GPU broker account.** Work with listed capacity or operate as a [provider](../provider/README.md) on nodes you already control.

## Prerequisites

- Reachable challenge URL (local, docker map, or Base proxy path)
- Hotkey + signing material for mutating routes
- Optional: Typer CLI via `uv run hypercluster`

## Typical flow

1. Probe identity

```bash
curl -fsS "$URL/health"
hypercluster health --url "$URL"
```

2. Browse offers

```bash
hypercluster marketplace offers list --url "$URL"
# or GET /v1/offers
```

3. Rent with transparent price/lifetime bounds

```bash
hypercluster marketplace rent --offer-id OFFER --url "$URL"
```

4. Submit a HyperJob (JSON/YAML/spec path)

```bash
hypercluster jobs submit --spec job.json --url "$URL"
hypercluster jobs status --id JOB --url "$URL"
```

5. Inspect scores / leaderboard

```bash
curl -fsS "$URL/v1/leaderboard"
hypercluster score show --hotkey YOUR_HOTKEY --url "$URL"
```

6. Terminate lease when finished

```bash
hypercluster marketplace terminate --lease-id LEASE --url "$URL"
```

## Tips

- Prefer digest-pinned images and honest fabric modes matching leased capacity (`ib` only if the offer actually requires and provides IB).
- Wall-clock timeouts protect the queue; efficiency scoring is compute-normalized, not pure latencies races.
- Unsigned or incomplete auth fails closed on writes.
- List endpoints without identity return empty private scopes rather than global dumps.

More detail: [Jobs](../jobs.md), [Marketplace](../marketplace.md), [Scoring](../scoring.md), [CLI](../cli.md).

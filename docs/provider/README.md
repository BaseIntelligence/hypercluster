# Provider guide (supply)

Providers advertise compute capacity into Hypercluster’s home-grown marketplace. Capacity can be self-owned SSH fleets, lab clusters, or hardware obtained outside the product (commercial rentals remain **ops-owned**, never a product SDK dependency).

## Prerequisites

- Challenge URL and signing hotkey
- Inventory facts: GPU model/count, SSH endpoints (secrets held out of git), optional TEE capability tags
- Fabric tools available on the host when claiming IB (or use sim nodes for local development)

## Typical flow

1. Register as provider and heartbeat.

```bash
# signed register / heartbeat via CLI or HTTP
hypercluster nodes register ... --url "$URL"
hypercluster nodes heartbeat ... --url "$URL"
```

2. Publish fabric reports when relevant.

```bash
hypercluster nodes fabric-scan --node-id NODE --url "$URL"
```

3. List an offer with **hard** `price_per_hour` and `max_lifetime_hours`.

```bash
hypercluster marketplace offer create --node-ids NODE --price 1.0 --lifetime 2 --url "$URL"
```

4. Keep nodes healthy while leased; active rentals are protected from casual idle reclamation.

5. Act on job runtime (provision/bind/results) according to your agent or manual worker path; post results with digests/proofs as required.

6. Withdraw or re-list offers after terminate.

```bash
hypercluster marketplace terminate --lease-id LEASE --url "$URL"   # if acting as owner policy
# DELETE /v1/offers/{id} to withdraw open listings
```

## Honesty expectations

- Do not claim InfiniBand when inventory reports zero IB devices.
- Inventory spoof and forbidden eth fallbacks zero fabric_gate / composite scoring.
- TEE claims without valid verification do not receive live tee_bonus.
- Self-deal volume between identical demand+supply hotkeys may be soft-penalized in aggregation.

## Local development

```bash
hypercluster sim seed --seed 0
hypercluster sim run-scenario --name marketplace --url http://127.0.0.1:3200
```

Sim inventory and launcher replace real SSH for CI. See [Fabric](../fabric.md) and [Live QA](../live-qa.md) for external single-GPU ops (maintainers).

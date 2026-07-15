# Marketplace

Hypercluster implements a **home-grown**, Lium-shaped marketplace entirely inside the challenge. Product code does not call commercial broker APIs. Capacity is whatever providers register (self fleets, lab clusters, or capacity a provider obtained out-of-band).

## Objects

```text
Provider
  └── Node[]                  # GPU hosts; optional multi-node fabric group
        └── FabricReport      # IB / topo / NCCL / NUMA self-report
Offer                         # sellable snapshot (single|cluster)
  └── Lease                   # time-bounded rental under price + lifetime caps
        └── Pod / cluster     # runtime binding (endpoints, status)
              └── Jobs
```

## HTTP routes

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/providers/register` | Provider hotkey onboarding |
| GET | `/v1/providers` | List providers |
| GET | `/v1/providers/me` | Caller provider view |
| POST | `/v1/providers/heartbeat` | Provider liveness |
| POST | `/v1/nodes` | Register / update node inventory |
| GET | `/v1/nodes` | List nodes (policy-filtered) |
| POST | `/v1/nodes/heartbeat` | Node heartbeat |
| GET | `/v1/nodes/{node_id}` | Node detail |
| POST | `/v1/nodes/{node_id}/fabric-scan` | Accept fabric scan |
| GET | `/v1/nodes/{node_id}/fabric-report` | Latest fabric report |
| POST | `/v1/offers` | Create offer |
| GET | `/v1/offers` | Browse marketplace |
| GET | `/v1/offers/{offer_id}` | Offer detail |
| DELETE | `/v1/offers/{offer_id}` | Withdraw listing |
| POST | `/v1/offers/{offer_id}/rent` | Create lease + pod |
| GET | `/v1/leases` | List leases (identity-scoped) |
| GET | `/v1/leases/{lease_id}` | Lease detail |
| POST | `/v1/leases/{lease_id}/terminate` | End early |
| GET | `/v1/pods/{pod_id}` | Pod status / endpoints |

Mutating routes require signed headers.

## Offer and rent rules

1. Offers must reference live **healthy** nodes. Cluster offers that require Fabrics must present consistent fabric when `require_ib`.
2. Hard guards: `price_per_hour` and `max_lifetime_hours` are always required.
3. Rent creates **Lease + Pod** in provisioning; provider marks readiness when SSH/agent is available.
4. Active rentals must not be killed by idle-only health logic (tenant short-circuit).
5. Terminate on renter request, expiry, or integrity-driven policy; free nodes and optionally re-list.

## Fail-closed list scoping

If a list endpoint scopes by optional `X-Hotkey` (leases, and similar filtered listings), **missing identity returns empty `items`**, never the full table.

## CLI

```bash
hypercluster marketplace offers list [--gpu-model MODEL]
hypercluster marketplace offer create --node-ids ... --price ... --lifetime ...
hypercluster marketplace rent --offer-id OFFER
hypercluster marketplace lease show --id LEASE
hypercluster marketplace terminate --lease-id LEASE
```

See [CLI](cli.md) and [Provider guide](provider/README.md).

## Local validation

```bash
hypercluster sim seed --seed 0
hypercluster sim run-scenario --name marketplace --url http://127.0.0.1:3200
```

Marketplace sim covers offer → rent → terminate and double-rent reject without cloud spend.

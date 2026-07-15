# Live QA protocol (single-GPU external path)

**Audience:** maintainers only.  
**Default CI and `pytest`:** never run live commercial rentals; never auto-load broker credentials.  
**Miners:** you do **not** need a commercial GPU broker account. Self-owned inventory through the home-grown marketplace is enough.

This protocol proves that Hypercluster's **product marketplace APIs** still accept a real single-GPU host when an operator provides capacity obtained out-of-band. It is **not** multi-node InfiniBand validation (that remains local sim).

## Hard boundaries

| Rule | Policy |
| --- | --- |
| Product package | No Verda/SDK/OAuth client; no `api.verda.com` adapter in `src/hypercluster` |
| Gated tests | Local unit/integration/sim only |
| Shape | **One** small **single-GPU** instance |
| Concurrency | Serial only |
| Duration | Smoke minutes-scale; few hours max inc. investigations |
| Spend | Hard rate + budget caps enforced in ops scripts |
| Cleanup | Always discontinue the reservation **and** terminate marketplace lease/pod |
| Secrets | Load only in the ops shell; never commit; never inject into the challenge process env |

## Tooling layout

External-only helpers live under `scripts/qa/` (outside the product adapter surface:

| Module | Role |
| --- | --- |
| `scripts/qa/verda_client.py` | Minimal urllib catalog/deploy/discontinue for ops shells |
| `scripts/qa/product_path.py` | Signed hypercluster marketplace register → heartbeat → offer → rent → job |
| `scripts/qa/verda_single_gpu_smoke.py` | Serial runner + evidence pack writer |
| Unit helpers | `tests/domain/test_m8_live_qa_helpers.py` (offline; no credentials) |

Default hard caps in the smoke runner are intentionally tight (rate and total budget). If caps would breach: **abort and discontinue**, do not expand scope.

## Operator sequence

1. **Start challenge API without commercial-cloud credentials in process env.**  
   Challenge must remain free of `VERDA_*` so product isolation audits hold.
2. **In a separate ops shell**, load external credentials from an **out-of-tree** file owned by the operator (mode-restricted). Do not write those values into the repo, docker-compose, or challenge settings.
3. **Run catalog availability at order time** (prices and stock drift). Prefer cheapest available single GPU that still can complete a short smoke.
4. **Deploy one instance only** via ops tooling (not product package imports used for production APIs).
5. **Register inventory through product path only:** provider register → node → heartbeat → fabric-scan as applicable → offer → rent.
6. **Submit a short job** on the leased pod through hypercluster jobs (single node). Multi-node IB claims are out of scope for this path.
7. **On success or failure:** terminate lease, discontinue cloud instance (prefer double-check), write evidence with ids, timestamps, rates, and upper-bound cost without raw OAuth secrets.
8. Stop local API processes when finished.

Example command shape (placeholders; adjust paths for your environment):

```bash
# API shell: no VERDA_*
export CHALLENGE_SHARED_TOKEN=dev-token
export CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////tmp/hypercluster-live/challenge.sqlite3
export HYPER_COMBINED_WORKER=true
mkdir -p /tmp/hypercluster-live
uv run hypercluster serve --host 127.0.0.1 --port 3200

# Ops shell: credentials loaded privately else where; never committed
uv run python scripts/qa/verda_single_gpu_smoke.py \
  --base-url http://127.0.0.1:3200 \
  --evidence-dir ./.docs-evidence/live-qa
```

Use a **gitignored** evidence directory (for example `.docs-evidence/`). Do not stage secrets.

## Evidence pack (expected contents)

| Artifact | Content |
| --- | --- |
| Product audit | Clean no-Verda product tree audit lines |
| Rental summary | Instance type, location, $/hr (no tokens) |
| Product ids | Provider/node/offer/lease/pod identifiers |
| Job terminal state | Job id and terminal status summary |
| Discontinue + cost | Confirmed tear-down and budget upper bound |

## What success does **not** claim

- Does not replace sim gates for multi-node fabric.
- Does not authorize embedding commercial SKUs into first-party product adapters.
- Does not change scoring formula or emission math.
- Does not require miners to create commercial broker accounts.

## Related

- [Security](security.md) for residual risks and secrets
- [Marketplace](marketplace.md) for product APIs used by the ops path
- [Fabric](fabric.md) for local multi-node validation

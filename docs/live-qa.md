# Live QA protocol (single-GPU external path)

**Audience:** maintainers only.  
**Default CI and `pytest`:** never run live commercial rentals; never auto-load broker credentials. GPU probe gates in CI use **FakeSsh only**.  
**Miners:** you do **not** need a commercial GPU broker account. Self-owned inventory through the home-grown marketplace is enough.

This protocol proves that Hypercluster's **product marketplace APIs** still accept a real single-GPU host when an operator provides capacity obtained out-of-band, and (optionally) that a **RealSSH GPU probe** yields `host_probe.json` / product evidence. It is **not** multi-node InfiniBand validation (that remains local sim). It does **not** authorize a product Verda adapter, does **not** call `set_weights`, and does **not** change the four-factor scoring formula.

## Hard boundaries

| Rule | Policy |
| --- | --- |
| Product package | No Verda/SDK/OAuth client; no `api.verda.com` adapter in `src/hypercluster` |
| Gated tests | Local unit/integration/sim + **FakeSsh GPU probe** only |
| Shape | **One** small **single-GPU** instance |
| Concurrency | Serial only |
| Duration | Smoke minutes-scale; few hours max inc. investigations |
| Spend | Hard rate + budget caps enforced in ops scripts |
| Cleanup | Always discontinue the reservation **and** terminate marketplace lease/pod |
| Secrets | Load only in the ops shell; never commit; never inject into the challenge process env |
| SSH keys | Ops file (`0600`); product stores fingerprint/ref only (never PEM) |
| Formula / chain | `correctness × efficiency × fabric_gate × tee_bonus` only; never `set_weights` |

## Tooling layout

External-only helpers live under `scripts/qa/` (outside the product adapter surface:

| Module | Role |
| --- | --- |
| `scripts/qa/verda_client.py` | Minimal urllib catalog/deploy/discontinue for ops shells |
| `scripts/qa/product_path.py` | Signed hypercluster marketplace register → heartbeat → offer → rent → job (+ optional product GPU probe attach) |
| `scripts/qa/host_gpu_probe.py` | WaitSSH + RealSsh allowlist probe → `host_probe.json` |
| `scripts/qa/verda_single_gpu_smoke.py` | Serial runner + evidence pack writer; `--with-host-probe` for M9 silicon bar |
| Unit helpers | `tests/domain/test_m8_live_qa_helpers.py`, `tests/domain/test_m9_host_gpu_probe_qa.py` (offline; FakeSsh / no credentials) |

Default hard caps in the smoke runner are intentionally tight (rate and total budget). If caps would breach: **abort and discontinue**, do not expand scope.

## Operator sequence

1. **Start challenge API without commercial-cloud credentials in process env.**  
   Challenge must remain free of `VERDA_*` so product isolation audits hold.
2. **In a separate ops shell**, load external credentials from an **out-of-tree** file owned by the operator (mode-restricted). Do not write those values into the repo, docker-compose, or challenge settings.
3. **Run catalog availability at order time** (prices and stock drift). Prefer cheapest available single GPU that still can complete a short smoke.
4. **Deploy one instance only** via ops tooling (not product package imports used for production APIs).
5. **Register inventory through product path only:** provider register → node → heartbeat → fabric-scan as applicable → offer → rent.
6. **Optional M9 host GPU probe (RealSSH):** after WaitSSH / register, run `host_gpu_probe.py` and/or product `POST /v1/nodes/{id}/probes/gpu` with `key_ref` (never PEM body). Pack must include `host_probe.json` with `status=passed` for silicon claim; sim job terminal alone is **not** silicon-green.
7. **Submit a short job** on the leased pod through hypercluster jobs (single node) if budget allows. Multi-node IB claims are out of scope for this path.
8. **On success or failure:** terminate lease, discontinue cloud instance (prefer double-check), write evidence with ids, timestamps, rates, and upper-bound cost without raw OAuth secrets or private keys.
9. Stop local API processes when finished.

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

# Optional M9 silicon bar (serial only; RealSSH key path on API + ops)
uv run python scripts/qa/verda_single_gpu_smoke.py \
  --base-url http://127.0.0.1:3200 \
  --with-host-probe \
  --ssh-key-file /path/to/operator_key.pem \
  --evidence-dir ./.docs-evidence/live-qa-m9
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
| M9 `host_probe.json` | `status=passed` + measured class (silicon bar when claim made) |
| M9 product evidence id | Non-null probe evidence when product path used |

## FakeSsh vs live (CI vs ops)

| Layer | Default `pytest` gate | Transport |
| --- | --- | --- |
| Domain pipeline, API, CLI, scoring integrity | Yes | **FakeSsh** fixtures |
| Allowlist / key redaction unit tests | Yes | No live sockets |
| Live rent + host RealSSH probe | **No** (serial ops) | RealSsh + external rent |

## What success does **not** claim

- Does not replace sim gates for multi-node fabric.
- Does not authorize embedding commercial SKUs into first-party product adapters.
- Does not change scoring formula or emission math (`set_weights` remains off-limits to the challenge).
- Does not require miners to create commercial broker accounts.
- Does not treat product sim job success alone as silicon-green without host probe evidence when M9 silicon is claimed.

## Related

- [GPU probe](gpu-probe.md) for ordered checks, FakeSsh, and security
- [Security](security.md) for residual risks and secrets
- [Marketplace](marketplace.md) for product APIs used by the ops path
- [Fabric](fabric.md) for local multi-node validation

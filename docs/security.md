# Security

Hypercluster aims for **cryptographically-anchored trust-but-audit**, not absolute guarantees for multi-node host honesty.

## Identity and auth

| Surface | Rule |
| --- | --- |
| Base identity | `/health`, `/ready`, `/version` are unauthenticated |
| Mutating public routes | Require miner signature headers: `X-Hotkey`, `X-Signature`, `X-Nonce`, `X-Timestamp` over a canonical body hash |
| Challenge internal | Shared challenge token for SDK-owned internals (for example weight GET) |
| List isolation | Identity-scoped list GETs that accept optional hotkey must fail closed to **empty items** when identity is missing (not "dump all") |
| Job single-resource GET | Status/attempt/fabric-report may be black-box poll surfaces without list visibility |

Incomplete or invalid signatures are rejected. CLI helpers must not print tokens.

## Secrets and configuration

| Item | Location | Rule |
| --- | --- | --- |
| Challenge shared token | env `CHALLENGE_SHARED_TOKEN` or `CHALLENGE_SHARED_TOKEN_FILE` | Prefer file mount; never commit real values |
| SQLite URL | `CHALLENGE_DATABASE_URL` under `/data` | Never `BASE_DATABASE_URL` |
| Product knobs | `HYPER_*` | Separate from Base `CHALLENGE_*` prefix |
| Provider SSH material | Operator-managed outside public payloads | Do not store plaintext secrets in git; GPU probe persists **fingerprint / key_ref only** (never PEM in SQLite or evidence JSON) |
| GPU probe FakeSsh | Tests / explicit allow only | Production must not silently fake silicon (`HYPER_SSH_TRANSPORT=real` default path fails closed without keys) |
| Commercial cloud QA secrets | Outside product process and outside default pytest | Never required for miners; never product Verda adapter |

Product documentation and `.env.example` use placeholders only.

## Data isolation

- Challenge SQLite on `/data` is the only durable store for marketplace/jobs/scores.
- Challenge never writes Base master Postgres and never executes validator `set_weights`.
- Raw weights are finite non-negative floats; NaN/Inf/negative weights are invalid.
- Outbound product policy denies commercial Verda control-plane hosts; optional maintainer tooling under `scripts/qa/` runs **outside** the challenge package import graph for product audits.

## Integrity and scoring

Integrity failures force **composite = 0** for an attempt, including (non-exhaustive): attestation fail, image/compose mutation, rank desync, inventory spoof, fabric lie, forbidden eth fallback when IB was required, GPU probe fatal fail / claim-vs-evidence mismatch under policy. Scoring formula stays four factors only (never `set_weights` from the challenge).

Self-deal and spam soft penalties can damp aggregated hotkey mass without inventing negative weights.

## TEE residual risk

- Offline fixtures prove the **verify machinery** and bonus wiring in CI. They do not prove live silicon.
- Live TEE (TDX / FPGA evidence paths) is optional and finish-closed whether hardware is absent.
- **Multi-node InfiniBand / RDMA is not encrypted by NVIDIA CC or TDX alone.** Do not market "encrypted fabric" from TEE tier alone. TEE strengthens collocated measurement when quotes bind the intended report_data layout.
- Sim-tier proofs must not mint live TEE bonuses.

## GPU probe security (non-TEE)

| Rule | Detail |
| --- | --- |
| Allowlist SSH only | `command_id` maps to fixed argv templates; reject unknown ids; no free-form remote shell from API bodies |
| No PEM storage | SQLite and public evidence expose at most key fingerprint / ref kind+name |
| Redaction | Evidence stores redacted short raw; secrets scrubbed from messages |
| FakeSsh boundary | Default gated CI uses FakeSsh fixtures; production refuses silent fake |
| Live cloud ops | External `scripts/qa/*` only; product package remains free of Verda SDK/OAuth |

See [GPU probe](gpu-probe.md) for the ordered fatal/advisory table.

## Residual risks (honest)

| Risk | Residual |
| --- | --- |
| Sophisticated multi-node host gaming without TEE | Mitigated by fabric gates, digests, optional TEE bonus, GPU probe inventory/UUID uniqueness, sample / honesty injects; not eliminated |
| Shared SQLite single-writer VI | Horizontal multi-replica challenge ownership deferred |
| Operator misconfig of master URL or tokens | Fail closed auth/push; never log secrets |
| Live cloud QA cost | Operator-enforced caps; always discontinue; not part of default CI |
| Open (non-sealed) microbench strength | Trades closed Lium `.so` hardness for reproducible open code; future AEAD challenge optional |

## Security checklist for operators

1. Mount challenge token as a file when possible (`CHALLENGE_SHARED_TOKEN_FILE`).
2. Keep `/data` volume private to the challenge process user.
3. Disable insecure signature modes outside local sim harnesses.
4. Point `HYPER_MASTER_BASE_URL` only at trusted master/mock endpoints.
5. Treat any commercial rental evidence as ops artifacts; keep credentials out of the product image and default process env.

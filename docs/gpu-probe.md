# GPU host probe (non-TEE silicon honesty)

Hypercluster can verify that a provider host has real GPU inventory and can perform a
minimal open CUDA smoke **without TEE**. The pipeline is Lium-inspired (ordered fatal
vs advisory SSH checks) and is **owned by the challenge**: no closed job provers,
no SN51 collateral markets, and **no commercial-cloud product adapter**.

## Formula and chain fences

```text
composite = correctness × efficiency × fabric_gate × tee_bonus
```

Probe failures and claim-vs-measured inventory mismatch feed **existing integrity
zeros** (correctness and/or `fabric_gate`). There is **no fifth published score factor**.
The challenge **never** calls on-chain `set_weights`.

## Ordered checks

| Check ID | Severity | Intent |
| --- | --- | --- |
| `ssh_connect` | fatal | SSH session to the node endpoint |
| `nvidia_smi_list` | fatal | At least one GPU from `nvidia-smi -L` |
| `gpu_count` | fatal | Count policy (non-zero, capped) |
| `gpu_model_match` | fatal | Claimed SKU/family matches measured name (normalize table) |
| `gpu_uuid_valid` | fatal | Well-formed GPU UUID set |
| `gpu_uuid_unique` | fatal | UUID not already claimed by another healthy/rented node |
| `vram_window` | fatal | Memory within model family window |
| `driver_present` | fatal | Driver / CUDA runtime strings present |
| `cuda_microbench` | fatal | Open CUDA microbench / digest smoke (`full` mode) |
| `docker_runtime` | fatal when docker required; else advisory | nvidia runtime visibility |
| `power_limit_ratio` | advisory | Soft power policy signal |
| `idle_util` | advisory | Soft idle-utilization signal |
| `fingerprint_stable` | fatal when prior verified set exists | UUID fingerprint churn forces re-admit |
| `claim_consistency` | fatal | Measured inventory vs node/offer claim |

Any **fatal** fail aborts the remaining pipeline and yields evidence `status=failed` or
`error`. **Advisory-only** fails still yield `status=passed` with advisories recorded.

## Transports

| Transport | When |
| --- | --- |
| **FakeSsh** | Default **gated CI** / tests (`HYPER_SSH_TRANSPORT=fake`, `HYPER_ALLOW_FAKE_SSH=true`) |
| **RealSsh** | Production and serial live ops; command **allowlist only** |

Production settings must **refuse silent FakeSsh**. Missing keys/host fail closed rather
than inventing GPUs.

## API

| Method | Path | Auth |
| --- | --- | --- |
| POST | `/v1/nodes/{id}/probes/gpu` | owner-signed; body may include `mode`, `timeout_s`, **`key_ref` only (never raw PEM)** |
| GET | `/v1/nodes/{id}/probes/gpu/latest` | black-box poll OK |
| GET | `/v1/nodes/{id}/probes/gpu/{evidence_id}` | black-box poll OK |
| GET | `/v1/nodes/{id}/probes/gpu` | list newest-first |
| GET | `/v1/evidence/gpu/{evidence_id}` | global evidence |
| POST | `/v1/nodes/{id}/evidence/gpu` | owner-signed external attach (digest checked) |

Node register does **not** auto-mark `gpu_probe_status=verified`.

## CLI

```bash
hypercluster nodes probe-gpu NODE_ID --mode full --json
hypercluster nodes probe-gpu-sim NODE_ID --pass-all --json
hypercluster nodes evidence list NODE_ID --json
hypercluster nodes evidence latest NODE_ID --json
hypercluster nodes evidence show EVIDENCE_ID --json
```

Exit codes: **0** passed, **2** failed checks, **3** transport/error, **1** usage.

## Security

| Rule | Detail |
| --- | --- |
| No private keys in SQLite | At most `key_fingerprint` / key_ref kind+name |
| No PEM in API bodies or evidence JSON | Operator keys live on disk (`0600`) or env |
| Allowlist SSH only | Unknown `command_id` abandoned; no free-form remote shell |
| Redacted raw | Response/store redaction + output caps |
| No Verda in product | Live cloud rent remains external `scripts/qa/*` only |

## Live maintainers path (optional)

Serial single-GPU ops may rent capacity **outside** the product package, register the host
via marketplace APIs, run a real SSH probe, and always discontinue under hard cost caps.
See [Live QA protocol](live-qa.md). Multi-node InfiniBand remains **local sim**.

## Related

- [Scoring](scoring.md) (formula + integrity)
- [Security](security.md)
- [CLI](cli.md)
- [Architecture](architecture.md)

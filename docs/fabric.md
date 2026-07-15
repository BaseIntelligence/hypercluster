# Fabric and NCCL (local sim + planner)

Hypercluster plans multi-node jobs against **FabricReport** inventory: IB devices, GPU topology digests, NUMA maps, NCCL version strings. Default CI honesty for multi-node paths is the **local simulator**, which exercises pack/spread placement and fabric gates without real HCAs.

## FabricReport contract

Minimum fields (conceptual):

| Field | Meaning |
| --- | --- |
| `node_id` | Node under report |
| `ib_devices` / `ib_rate_gbps` | IB capability summary |
| `gpu_topo_sha256` | Hash of `nvidia-smi`-style topo text |
| `numa_map` | CPU/GPU NUMA placement |
| `nccl_version` | Reported NCCL stack |
| `report_digest` | Canonical sha256 of the report |

Admission: jobs/offers requiring IB fail closed when reports show zero IB devices.

## Planner

| Input | Output |
| --- | --- |
| world size, nnodes, nproc_per_node, policy, fabric mode, node reports | rankmap, `nccl_env`, planner version, graph digest |

Policies:

- **pack**: fill fewest nodes (NVLink-dense allreduce friendly)
- **spread**: distribute ranks for fabric-stress scenarios

Representative env keys emitted into placements:

```text
MASTER_ADDR, MASTER_PORT
NCCL_SOCKET_IFNAME
NCCL_IB_HCA, NCCL_IB_GID_INDEX
NCCL_NET=IB when fabric=ib
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

## Launcher contract

```text
torchrun-style multi-node fan-out:
  --nnodes --nproc_per_node --node_rank --rdzv_endpoint ...
```

Results carry metrics, digests, optional NCCL debug digests, and status `succeeded` / `failed` / `timeout`.

## Honesty layers (multi-node without full re-exec)

| Layer | Mechanism |
| --- | --- |
| L0 | Digest-pinned image + entrypoint allowlist |
| L1 | Synthetic fixed-size NCCL allreduce tolerance band |
| L2 | Cross-rank progress digests |
| L3 | Optional per-node TEE quote bound to job digests |
| L4 | Cheap single-node checks / rare sample audits |

**Limitation:** GPU confidential computing and TDX do not encrypt multi-node InfiniBand. Fabric gates catch forbidden fallbacks; they do not invent wire confidentiality.

## Sim injects (local/dev)

Product knobs (never for silent production gaming):

```text
HYPER_SIM_INVENTORY_SPOOF=false
HYPER_SIM_ETH_FALLBACK=false
HYPER_SIM_HONESTY_LEVEL=l1
HYPER_SIM_LAUNCH_FAIL=false
HYPER_SIM_LAUNCH_TIMEOUT=false
```

## CLI

```bash
hypercluster nodes fabric-scan --node-id NODE
hypercluster fabric plan --spec placement-spec.json    # dry-run only
hypercluster fabric report show --job-id JOB
# fabric launch is a gated / dev-only path and may deny by policy
```

## Local validation

```bash
hypercluster sim seed --seed 0
hypercluster sim run-scenario --name nccl --url http://127.0.0.1:3200
hypercluster sim doctor --offline
```

`nccl` scenario covers multi-node pack/spread plus fabric_gate fail injection.

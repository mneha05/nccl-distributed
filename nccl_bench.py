from __future__ import annotations

import argparse
import json
import os
import socket
from multiprocessing import get_context

import numpy as np
import torch
import torch.distributed as dist


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def worker(rank: int, world_size: int, elements: int, warmup: int, iterations: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    x = torch.full((elements,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32)
    expected = float(world_size * (world_size + 1) // 2)

    for _ in range(warmup):
        dist.all_reduce(x)
        x.fill_(float(rank + 1))
    torch.cuda.synchronize(rank)
    dist.barrier()

    times = []
    for _ in range(iterations):
        x.fill_(float(rank + 1))
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(x)
        stop.record()
        torch.cuda.synchronize(rank)
        times.append(float(start.elapsed_time(stop)))

        if not torch.allclose(x[0], torch.tensor(expected, device=x.device)):
            raise RuntimeError(f"rank {rank}: incorrect AllReduce result {x[0].item()} != {expected}")

    if rank == 0:
        a = np.asarray(times, dtype=np.float64)
        bytes_per_tensor = elements * 4
        # Common AllReduce algorithmic bandwidth convention.
        alg_bw = (bytes_per_tensor / (a.mean() / 1000.0)) / 1e9
        queue.put({
            "world_size": world_size,
            "elements": elements,
            "iterations": iterations,
            "mean_ms": round(float(a.mean()), 4),
            "p50_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "algorithmic_bandwidth_GBps": round(float(alg_bw), 3),
        })

    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--elements", type=int, default=8_388_608)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"need {args.world_size} GPUs, found {torch.cuda.device_count()}")

    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(
            target=worker,
            args=(rank, args.world_size, args.elements, args.warmup, args.iterations, port, queue),
        )
        for rank in range(args.world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(proc.exitcode)

    print(json.dumps(queue.get(), indent=2))


if __name__ == "__main__":
    main()

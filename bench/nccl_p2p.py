#!/usr/bin/env python3
"""Measure NCCL point-to-point bandwidth between the two nodes.

This is the number §08's pipeline-parallel option rests on. That section claims
a 70B microbatch cut costs "roughly 67 MB, about a quarter second on 2.5GbE
against ~15 s of compute" -- a ~60:1 compute-to-comms ratio that makes the
pipeline bubble, not the wire, the limiting factor. If the real ratio is nearer
10:1 the economics change and the option is worth less than it looks.

Pipeline parallelism uses send/recv, not all-reduce: with two stages there is one
cut, crossed once forward with activations and once backward with gradients. So
P2P is the primitive to measure, and a microbatch costs two crossings.

  rank 0 (desktop) $ ./nccl_p2p.py --rank 0 --addr 10.10.0.1
  rank 1 (laptop)  $ ./nccl_p2p.py --rank 1 --addr 10.10.0.1
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

SIZES_MB = [0.015625, 0.0625, 0.25, 1, 16, 67]
ITERS = 50
WARMUP = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--addr", default="10.10.0.1")
    ap.add_argument("--port", default="29555")
    ap.add_argument("--backend", default="nccl")
    args = ap.parse_args()

    os.environ["MASTER_ADDR"] = args.addr
    os.environ["MASTER_PORT"] = args.port

    dist.init_process_group(backend=args.backend, rank=args.rank, world_size=2)
    torch.cuda.set_device(0)
    peer = 1 - args.rank

    if args.rank == 0:
        print(f"backend {args.backend}  device {torch.cuda.get_device_name()}", flush=True)
        print(f"{'payload':>10}{'latency':>12}{'bandwidth':>14}{'per microbatch':>16}", flush=True)

    for mb in SIZES_MB:
        buf = torch.empty(max(1, int(mb * 1024 * 1024) // 2), dtype=torch.bfloat16, device="cuda")
        for _ in range(WARMUP):
            if args.rank == 0:
                dist.send(buf, peer)
                dist.recv(buf, peer)
            else:
                dist.recv(buf, peer)
                dist.send(buf, peer)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(ITERS):
            if args.rank == 0:
                dist.send(buf, peer)
                dist.recv(buf, peer)
            else:
                dist.recv(buf, peer)
                dist.send(buf, peer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # One round trip is two crossings: the forward activation and the
        # backward gradient for the same microbatch.
        one_way = elapsed / (ITERS * 2)
        if args.rank == 0:
            print(f"{mb*1024:>8.0f} KB{one_way*1000:>10.2f}ms{mb/one_way:>11.0f} MB/s"
                  f"{one_way*2*1000:>13.2f}ms", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

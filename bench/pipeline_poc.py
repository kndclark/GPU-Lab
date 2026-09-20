#!/usr/bin/env python3
"""Two-stage pipeline-parallel training across the two nodes.

The point is the mechanic that DeepSpeed and torch.distributed.pipelining
automate, done by hand so it can be inspected: autograd does not cross a process
boundary, so the pipeline cut has to be stitched manually.

  stage 0   forward -> send activations (detached)
  stage 1   recv -> requires_grad_() -> forward -> loss -> backward
            -> the input's .grad is the gradient of the cut
  stage 0   recv that gradient and use it to seed its own backward

Get that wrong and the loss still moves -- it just optimises the wrong thing --
so this reports a single-process reference loss on the same data and seed. Two
curves that track each other is the only evidence the cut is correct.

  rank 0 (laptop)  $ ./pipeline_poc.py --rank 0 --addr 10.10.0.1
  rank 1 (desktop) $ ./pipeline_poc.py --rank 1 --addr 10.10.0.1
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Stage(nn.Module):
    """Half the model. Stage 0 owns the embedding, stage 1 owns the head."""

    def __init__(self, dim, heads, layers, vocab, first, last):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim) if first else None
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab) if last else None

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        if self.head is not None:
            x = self.head(x)
        return x


def reference_loss(args, device):
    """Same data, same seed, one process -- the curve the pipeline must match."""
    torch.manual_seed(args.seed)
    whole = nn.Sequential().to(device)
    torch.manual_seed(args.seed)
    s0 = Stage(args.dim, args.heads, args.layers, args.vocab, True, False).to(device)
    torch.manual_seed(args.seed + 1)
    s1 = Stage(args.dim, args.heads, args.layers, args.vocab, False, True).to(device)
    opt = torch.optim.Adam(list(s0.parameters()) + list(s1.parameters()), lr=args.lr)
    losses = []
    g = torch.Generator(device="cpu").manual_seed(args.seed + 99)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.microbatches):
            ids = torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device)
            tgt = torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device)
            out = s1(s0(ids))
            loss = nn.functional.cross_entropy(out.reshape(-1, args.vocab), tgt.reshape(-1))
            (loss / args.microbatches).backward()
            losses.append(loss.item())
        opt.step()
    return sum(losses[-args.microbatches:]) / args.microbatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--addr", default="10.10.0.1")
    ap.add_argument("--port", default="29556")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4, help="per stage")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", action="store_true",
                    help="also run both stages in this process, to compare curves")
    args = ap.parse_args()

    os.environ["MASTER_ADDR"] = args.addr
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group(backend="nccl", rank=args.rank, world_size=2)
    device = "cuda"
    torch.cuda.set_device(0)
    peer = 1 - args.rank
    first = args.rank == 0

    torch.manual_seed(args.seed if first else args.seed + 1)
    stage = Stage(args.dim, args.heads, args.layers, args.vocab, first, not first).to(device)
    opt = torch.optim.Adam(stage.parameters(), lr=args.lr)

    act = (args.batch, args.seq, args.dim)
    if first:
        params = sum(p.numel() for p in stage.parameters())
        print(f"node {torch.cuda.get_device_name()}  stage params {params/1e6:.1f}M", flush=True)
        print(f"cut payload {args.batch*args.seq*args.dim*2/1024**2:.2f} MiB per microbatch", flush=True)
        print(f"{'step':>5}{'loss':>10}{'compute':>11}{'comms':>10}{'ratio':>9}", flush=True)

    # The task has to be learnable or the loss curve proves nothing: with random
    # targets the only thing to learn is the unigram prior, so loss falls to
    # ln(vocab) = 9.01 here whether or not gradients actually cross the cut.
    # Shift-by-one is deterministic, so loss well below ln(vocab) is only
    # reachable if the backward pass genuinely spans both stages.
    floor = torch.log(torch.tensor(float(args.vocab))).item()
    if first:
        print(f"task: reconstruct the input ids   chance loss = ln(vocab) = {floor:.2f}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed + 99)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        t_compute = t_comms = 0.0
        last_loss = None

        for _ in range(args.microbatches):
            if first:
                ids = torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device)
                t0 = time.perf_counter(); torch.cuda.synchronize()
                out = stage(ids)
                torch.cuda.synchronize(); t_compute += time.perf_counter() - t0

                t0 = time.perf_counter()
                dist.send(out.detach().contiguous(), peer)
                grad = torch.empty(act, device=device)
                dist.recv(grad, peer)
                torch.cuda.synchronize(); t_comms += time.perf_counter() - t0

                t0 = time.perf_counter()
                out.backward(grad)
                torch.cuda.synchronize(); t_compute += time.perf_counter() - t0
            else:
                buf = torch.empty(act, device=device)
                t0 = time.perf_counter()
                dist.recv(buf, peer)
                torch.cuda.synchronize(); t_comms += time.perf_counter() - t0

                t0 = time.perf_counter(); torch.cuda.synchronize()
                inp = buf.detach().requires_grad_(True)
                # Same seed and draw order as stage 0, so these are the very ids
                # stage 0 embedded; the target is those ids shifted by one.
                ids = torch.randint(0, args.vocab, (args.batch, args.seq), generator=g)
                tgt = ids.to(device)
                out = stage(inp)
                loss = nn.functional.cross_entropy(out.reshape(-1, args.vocab), tgt.reshape(-1))
                (loss / args.microbatches).backward()
                torch.cuda.synchronize(); t_compute += time.perf_counter() - t0
                last_loss = loss.item()

                t0 = time.perf_counter()
                dist.send(inp.grad.contiguous(), peer)
                torch.cuda.synchronize(); t_comms += time.perf_counter() - t0

        opt.step()

        stats = torch.tensor([t_compute, t_comms, last_loss if last_loss else 0.0], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if first:
            c, m, l = stats.tolist()
            print(f"{step:>5}{l:>10.4f}{c:>10.2f}s{m:>9.2f}s{c/max(m,1e-9):>8.1f}:1", flush=True)

    if first and args.reference:
        ref = reference_loss(args, device)
        print(f"\nsingle-process reference final loss: {ref:.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

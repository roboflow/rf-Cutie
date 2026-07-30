"""Standalone MPS VRAM bench for the memory-read hot path.

Not a pytest assertion — MPS allocator memory numbers are noisy run-to-run, so
this is evidence to eyeball (before/after the in-place-op refactor), not a CI
gate. Correctness is covered by tests/test_mps_parity.py; this script only
measures peak Apple-GPU memory and wall-clock for cutie.model.utils.memory_utils
get_similarity + do_softmax at a realistic memory-bank size.

Usage:
    python scripts/bench_vram_mps.py [--frames N] [--hw N] [--ck N] [--iters N]
"""

import argparse
import time

import torch
from torch import mps

from cutie.model.utils.memory_utils import do_softmax, get_similarity


def _build_inputs(
    *, batch: int, ck: int, num_memory_frames: int, hw: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = num_memory_frames * hw
    mk = torch.randn(batch, ck, n, device=device)
    ms = torch.rand(batch, 1, n, device=device)
    qk = torch.randn(batch, ck, hw, device=device)
    qe = torch.rand(batch, ck, hw, device=device)
    return mk, ms, qk, qe


def _run_once(mk: torch.Tensor, ms: torch.Tensor, qk: torch.Tensor, qe: torch.Tensor) -> torch.Tensor:
    similarity = get_similarity(mk, ms, qk, qe)
    return do_softmax(similarity)


def bench(*, batch: int, ck: int, num_memory_frames: int, hw: int, iters: int) -> None:
    if iters < 1:
        raise ValueError('iters must be >= 1')
    if not torch.backends.mps.is_available():
        print('MPS not available on this machine — nothing to bench.')
        return

    device = 'mps'
    mk, ms, qk, qe = _build_inputs(batch=batch, ck=ck, num_memory_frames=num_memory_frames, hw=hw, device=device)

    # warm up (first MPS dispatch pays kernel-compile cost, not representative)
    _run_once(mk, ms, qk, qe)
    torch.mps.synchronize()

    mps.empty_cache()
    baseline_allocated = mps.current_allocated_memory()

    start = time.perf_counter()
    for _ in range(iters):
        affinity = _run_once(mk, ms, qk, qe)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    peak_allocated = mps.driver_allocated_memory()

    print('MPS VRAM bench — cutie.model.utils.memory_utils (get_similarity + do_softmax)')
    print(f'  shapes: batch={batch} ck={ck} memory_frames={num_memory_frames} hw={hw} -> N={num_memory_frames * hw}')
    print(f'  iters: {iters}')
    print(f'  baseline allocated (post-warmup, pre-loop): {baseline_allocated / 2**20:.2f} MiB')
    print(f'  driver allocated (peak, post-loop):         {peak_allocated / 2**20:.2f} MiB')
    print(f'  wall-clock: {elapsed:.4f}s total, {elapsed / iters * 1000:.3f}ms/iter')
    print(f'  output shape: {tuple(affinity.shape)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--ck', type=int, default=64, help='key channel dim')
    parser.add_argument('--frames', type=int, default=20, dest='num_memory_frames', help='accumulated memory frames')
    parser.add_argument('--hw', type=int, default=30 * 54, help='flattened spatial size (H*W/patch)')
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    bench(batch=args.batch, ck=args.ck, num_memory_frames=args.num_memory_frames, hw=args.hw, iters=args.iters)


if __name__ == '__main__':
    main()

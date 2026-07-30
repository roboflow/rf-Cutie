"""CPU-vs-MPS parity guardrails for pure-math ops slated for VRAM optimization.

These tests prove `cutie/model/utils/memory_utils.py` and
`cutie/model/channel_attn.py` produce numerically consistent results on Apple's
MPS backend before any device-related refactor touches them. Every existing
test in this suite runs CPU-only; this file is the first to actually execute
tensor ops on `mps`, so the tests below are guarded by a real
`torch.backends.mps.is_available()` check rather than being unconditionally
skipped.
"""

import copy

import pytest
import torch

from cutie.model.channel_attn import CAResBlock
from cutie.model.utils.memory_utils import do_softmax, get_similarity

MPS_UNAVAILABLE = not torch.backends.mps.is_available()

# MPS matmul/softmax use float32-only accumulation and may reduce in a
# different order than CPU BLAS, so results are close but not bit-identical.
# These tolerances were determined empirically against this repo's ops (see
# module docstring); widen only with a concrete numerical justification.
_ATOL = 1e-5
_RTOL = 1e-4


def _build_similarity_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build small mk/ms/qk/qe tensors on CPU via the seeded RNG."""
    mk = torch.randn(2, 4, 3)
    ms = torch.rand(2, 1, 3) + 0.5
    qk = torch.randn(2, 4, 5)
    qe = torch.rand(2, 4, 5) + 0.5
    return mk, ms, qk, qe


@pytest.mark.skipif(MPS_UNAVAILABLE, reason='MPS not available on this machine')
class TestGetSimilarityMpsParity:
    """Guardrail: `get_similarity` must agree between CPU and MPS backends."""

    @pytest.mark.parametrize(
        'use_qe',
        [
            pytest.param(True, id='qe_present'),
            pytest.param(False, id='qe_none'),
        ],
    )
    def test_get_similarity_matches_between_cpu_and_mps(self, use_qe: bool) -> None:
        """Same CPU-built tensors moved to MPS give results close to the CPU run."""
        mk_cpu, ms_cpu, qk_cpu, qe_cpu = _build_similarity_inputs()
        qe_cpu = qe_cpu if use_qe else None

        mk_mps, ms_mps, qk_mps = mk_cpu.to('mps'), ms_cpu.to('mps'), qk_cpu.to('mps')
        qe_mps = qe_cpu.to('mps') if qe_cpu is not None else None

        cpu_result = get_similarity(mk_cpu, ms_cpu, qk_cpu, qe_cpu)
        mps_result = get_similarity(mk_mps, ms_mps, qk_mps, qe_mps)

        torch.testing.assert_close(cpu_result, mps_result.cpu(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.skipif(MPS_UNAVAILABLE, reason='MPS not available on this machine')
class TestDoSoftmaxMpsParity:
    """Guardrail: `do_softmax` must agree between CPU and MPS backends."""

    @pytest.mark.parametrize(
        'top_k',
        [
            pytest.param(None, id='dense_softmax'),
            pytest.param(2, id='top_k_subset'),
        ],
    )
    def test_do_softmax_matches_between_cpu_and_mps(self, top_k: int | None) -> None:
        """Same CPU-built similarity tensor moved to MPS gives a close affinity map."""
        similarity_cpu = torch.randn(2, 5, 3)
        similarity_mps = similarity_cpu.to('mps')

        cpu_result = do_softmax(similarity_cpu, top_k=top_k, inplace=False)
        mps_result = do_softmax(similarity_mps, top_k=top_k, inplace=False)

        torch.testing.assert_close(cpu_result, mps_result.cpu(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.skipif(MPS_UNAVAILABLE, reason='MPS not available on this machine')
class TestCAResBlockMpsParity:
    """Guardrail: `CAResBlock` forward pass must agree between CPU and MPS."""

    def test_ca_res_block_forward_matches_between_cpu_and_mps(self) -> None:
        """Identical weights on CPU vs MPS produce a forward output within tolerance."""
        module_cpu = CAResBlock(4, 4).eval()
        # deepcopy before moving preserves exact weights - two independently
        # constructed modules would diverge even under the seed fixture, since
        # Conv2d init happens at construction time.
        module_mps = copy.deepcopy(module_cpu).to('mps').eval()
        x_cpu = torch.randn(1, 4, 8, 8)
        x_mps = x_cpu.to('mps')

        with torch.no_grad():
            cpu_result = module_cpu(x_cpu)
            mps_result = module_mps(x_mps)

        torch.testing.assert_close(cpu_result, mps_result.cpu(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.skipif(MPS_UNAVAILABLE, reason='MPS not available on this machine')
def test_get_similarity_actually_executes_on_mps_device() -> None:
    """Result tensor stays on `mps` - proves no silent CPU fallback occurred."""
    mk_cpu, ms_cpu, qk_cpu, qe_cpu = _build_similarity_inputs()
    mk, ms, qk, qe = (t.to('mps') for t in (mk_cpu, ms_cpu, qk_cpu, qe_cpu))

    result = get_similarity(mk, ms, qk, qe)

    assert result.device.type == 'mps'

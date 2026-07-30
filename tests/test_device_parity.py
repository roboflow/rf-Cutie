"""CPU-vs-accelerator parity guardrails for pure-math ops slated for VRAM optimization.

These tests prove `cutie/model/utils/memory_utils.py` and
`cutie/model/channel_attn.py` produce numerically consistent results on every
GPU backend PyTorch supports on this codebase's target platforms - CUDA and
Apple's MPS - before any device-related refactor touches them. Every existing
test in this suite otherwise runs CPU-only; this file is the first to actually
execute tensor ops on an accelerator, so each backend's tests are guarded by a
real `is_available()` check rather than being unconditionally skipped. A
machine with only one backend (e.g. this MacBook has MPS, no CUDA) still runs
that backend's cases for real and simply skips the other - both are exercised
wherever hardware allows, neither is assumed absent.
"""

import copy

import pytest
import torch

from cutie.model.channel_attn import CAResBlock
from cutie.model.utils.memory_utils import do_softmax, get_similarity

# Backends to check parity against CPU, each independently skipped when its
# hardware isn't present - never assume one backend stands in for the other.
_ACCELERATOR_DEVICES = [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available on this machine'),
    ),
    pytest.param(
        'mps',
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS not available on this machine'),
    ),
]

# Both CUDA and MPS accumulate in float32 and may reduce in a different order
# than CPU BLAS, so results are close but not bit-identical. These tolerances
# were determined empirically against this repo's ops (see module docstring);
# widen only with a concrete numerical justification.
_ATOL = 1e-5
_RTOL = 1e-4


def _build_similarity_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build small mk/ms/qk/qe tensors on CPU via the seeded RNG."""
    mk = torch.randn(2, 4, 3)
    ms = torch.rand(2, 1, 3) + 0.5
    qk = torch.randn(2, 4, 5)
    qe = torch.rand(2, 4, 5) + 0.5
    return mk, ms, qk, qe


class TestGetSimilarityDeviceParity:
    """Guardrail: `get_similarity` must agree between CPU and each accelerator backend."""

    @pytest.mark.parametrize('device', _ACCELERATOR_DEVICES)
    @pytest.mark.parametrize(
        'use_qe',
        [
            pytest.param(True, id='qe_present'),
            pytest.param(False, id='qe_none'),
        ],
    )
    def test_get_similarity_matches_between_cpu_and_device(self, device: str, use_qe: bool) -> None:
        """Same CPU-built tensors moved to the accelerator give results close to the CPU run."""
        mk_cpu, ms_cpu, qk_cpu, qe_cpu = _build_similarity_inputs()
        qe_cpu = qe_cpu if use_qe else None

        mk_dev, ms_dev, qk_dev = mk_cpu.to(device), ms_cpu.to(device), qk_cpu.to(device)
        qe_dev = qe_cpu.to(device) if qe_cpu is not None else None

        cpu_result = get_similarity(mk_cpu, ms_cpu, qk_cpu, qe_cpu)
        device_result = get_similarity(mk_dev, ms_dev, qk_dev, qe_dev)

        torch.testing.assert_close(cpu_result, device_result.cpu(), atol=_ATOL, rtol=_RTOL)


class TestDoSoftmaxDeviceParity:
    """Guardrail: `do_softmax` must agree between CPU and each accelerator backend."""

    @pytest.mark.parametrize('device', _ACCELERATOR_DEVICES)
    @pytest.mark.parametrize(
        'top_k',
        [
            pytest.param(None, id='dense_softmax'),
            pytest.param(2, id='top_k_subset'),
        ],
    )
    def test_do_softmax_matches_between_cpu_and_device(self, device: str, top_k: int | None) -> None:
        """Same CPU-built similarity tensor moved to the accelerator gives a close affinity map."""
        similarity_cpu = torch.randn(2, 5, 3)
        similarity_dev = similarity_cpu.to(device)

        cpu_result = do_softmax(similarity_cpu, top_k=top_k, inplace=False)
        device_result = do_softmax(similarity_dev, top_k=top_k, inplace=False)

        torch.testing.assert_close(cpu_result, device_result.cpu(), atol=_ATOL, rtol=_RTOL)


class TestCAResBlockDeviceParity:
    """Guardrail: `CAResBlock` forward pass must agree between CPU and each accelerator backend."""

    @pytest.mark.parametrize('device', _ACCELERATOR_DEVICES)
    def test_ca_res_block_forward_matches_between_cpu_and_device(self, device: str) -> None:
        """Identical weights on CPU vs the accelerator produce a forward output within tolerance."""
        module_cpu = CAResBlock(4, 4).eval()
        # deepcopy before moving preserves exact weights - two independently
        # constructed modules would diverge even under the seed fixture, since
        # Conv2d init happens at construction time.
        module_dev = copy.deepcopy(module_cpu).to(device).eval()
        x_cpu = torch.randn(1, 4, 8, 8)
        x_dev = x_cpu.to(device)

        with torch.no_grad():
            cpu_result = module_cpu(x_cpu)
            device_result = module_dev(x_dev)

        torch.testing.assert_close(cpu_result, device_result.cpu(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize('device', _ACCELERATOR_DEVICES)
def test_get_similarity_actually_executes_on_device(device: str) -> None:
    """Result tensor stays on the accelerator - proves no silent CPU fallback occurred."""
    mk_cpu, ms_cpu, qk_cpu, qe_cpu = _build_similarity_inputs()
    mk, ms, qk, qe = (t.to(device) for t in (mk_cpu, ms_cpu, qk_cpu, qe_cpu))

    result = get_similarity(mk, ms, qk, qe)

    assert result.device.type == device

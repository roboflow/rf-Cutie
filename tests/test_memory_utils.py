"""Tests for deterministic memory similarity, affinity, and readout math.

The tests below characterize the CURRENT behavior of
``cutie/model/utils/memory_utils.py`` as a safety net ahead of an in-place-op
VRAM refactor. They pin exact numeric parity (not just finiteness) so any
future in-place rewrite can be checked against these baselines.
"""

import math

import pytest
import torch

from cutie.model.utils.memory_utils import do_softmax, get_affinity, get_similarity, readout


def test_cpu_rng_starts_from_the_suite_seed() -> None:
    """The autouse fixture resets Torch randomness to the declared test seed."""
    values = torch.rand(4)

    torch.manual_seed(7)

    assert torch.equal(values, torch.rand(4))


def test_similarity_and_affinity_are_finite_and_normalized() -> None:
    """Similarity variants produce normalized per-query affinity weights."""
    memory_key = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    memory_shrinkage = torch.ones((1, 1, 2))
    query_key = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    query_selection = torch.ones((1, 2, 2))

    selected = get_similarity(memory_key, memory_shrinkage, query_key, query_selection)
    unselected = get_similarity(memory_key, memory_shrinkage, query_key, None)
    affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)

    assert selected.shape == unselected.shape == (1, 2, 2)
    assert torch.isfinite(selected).all()
    assert torch.isfinite(unselected).all()
    torch.testing.assert_close(affinity.sum(dim=1), torch.ones((1, 2)))


def test_top_k_softmax_and_readout_keep_only_selected_memory_entries() -> None:
    """Top-k affinity is sparse and readout follows its selected memory values."""
    similarity = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]]])

    affinity, usage = do_softmax(similarity, top_k=1, return_usage=True)

    assert torch.equal(affinity, torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]))
    assert torch.equal(usage, torch.tensor([[0.0, 1.0, 1.0]]))
    memory_values = torch.tensor([[[[[10.0, 20.0]]]]])
    assert torch.equal(readout(affinity[:, :2], memory_values), torch.tensor([[[[20.0, 0.0]]]]))


def _reference_similarity(
    mk: torch.Tensor,
    ms: torch.Tensor | None,
    qk: torch.Tensor,
    qe: torch.Tensor | None,
) -> torch.Tensor:
    """Elementwise reimplementation of get_similarity's formula, independent of its matmul path."""
    b_dim, ck_dim, n_dim = mk.shape
    hw_dim = qk.shape[2]
    out = torch.zeros(b_dim, n_dim, hw_dim, dtype=mk.dtype)
    for b in range(b_dim):
        for n in range(n_dim):
            for hw in range(hw_dim):
                a_sq = 0.0
                two_ab = 0.0
                b_sq = 0.0
                for ck in range(ck_dim):
                    mk_val = mk[b, ck, n].item()
                    qk_val = qk[b, ck, hw].item()
                    if qe is not None:
                        qe_val = qe[b, ck, hw].item()
                        a_sq += (mk_val**2) * qe_val
                        two_ab += 2 * mk_val * qk_val * qe_val
                        b_sq += qe_val * (qk_val**2)
                    else:
                        a_sq += mk_val**2
                        two_ab += 2 * mk_val * qk_val
                similarity = -a_sq + two_ab - b_sq
                similarity /= math.sqrt(ck_dim)
                if ms is not None:
                    similarity *= ms[b, 0, n].item()
                out[b, n, hw] = similarity
    return out


@pytest.mark.parametrize(
    'has_qe, has_ms, ck_dim, n_dim',
    [
        pytest.param(True, True, 2, 2, id='qe-present_ms-present_ck2_n2'),
        pytest.param(True, True, 1, 2, id='qe-present_ms-present_ck1-boundary'),
        pytest.param(True, True, 2, 1, id='qe-present_ms-present_n1-single-key'),
        pytest.param(False, True, 2, 2, id='qe-none_ms-present'),
        pytest.param(True, False, 2, 2, id='qe-present_ms-none'),
        pytest.param(False, False, 2, 2, id='qe-none_ms-none'),
    ],
)
def test_get_similarity_matches_independent_reference_implementation(
    has_qe: bool,
    has_ms: bool,
    ck_dim: int,
    n_dim: int,
) -> None:
    """get_similarity matches an elementwise reference across qe/ms presence and CK/N boundaries."""
    hw_dim = 2
    mk = torch.randn(1, ck_dim, n_dim)
    ms = torch.rand(1, 1, n_dim) + 0.5 if has_ms else None
    qk = torch.randn(1, ck_dim, hw_dim)
    qe = torch.rand(1, ck_dim, hw_dim) + 0.1 if has_qe else None

    similarity = get_similarity(mk, ms, qk, qe)

    expected = _reference_similarity(mk, ms, qk, qe)
    torch.testing.assert_close(similarity, expected)


def test_get_similarity_add_batch_dim_matches_manual_unsqueeze() -> None:
    """add_batch_dim=True on un-batched tensors matches manually unsqueezing then calling normally."""
    mk = torch.randn(2, 2)
    ms = torch.rand(1, 2) + 0.5
    qk = torch.randn(2, 2)
    qe = torch.rand(2, 2) + 0.1

    via_flag = get_similarity(mk, ms, qk, qe, add_batch_dim=True)
    via_manual_unsqueeze = get_similarity(mk.unsqueeze(0), ms.unsqueeze(0), qk.unsqueeze(0), qe.unsqueeze(0))

    torch.testing.assert_close(via_flag, via_manual_unsqueeze)


def test_do_softmax_dense_branch_matches_torch_softmax() -> None:
    """do_softmax with top_k=None reduces to a plain softmax over the memory dimension."""
    similarity = torch.randn(2, 3, 4)

    affinity = do_softmax(similarity, top_k=None)

    torch.testing.assert_close(affinity, torch.softmax(similarity, dim=1))


def test_do_softmax_top_k_equals_full_size_matches_dense_softmax() -> None:
    """Requesting top_k equal to the memory size reduces to full dense softmax regardless of ordering."""
    similarity = torch.randn(1, 3, 2)

    affinity = do_softmax(similarity, top_k=3)

    torch.testing.assert_close(affinity, torch.softmax(similarity, dim=1))


def test_do_softmax_inplace_flag_controls_tensor_identity() -> None:
    """inplace=True mutates and returns the input tensor; inplace=False returns an equal-valued new tensor."""
    base_similarity = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]]])
    similarity_for_inplace = base_similarity.clone()
    similarity_for_out_of_place = base_similarity.clone()

    affinity_inplace = do_softmax(similarity_for_inplace, top_k=1, inplace=True)
    affinity_out_of_place = do_softmax(similarity_for_out_of_place, top_k=1, inplace=False)

    assert affinity_inplace is similarity_for_inplace
    assert affinity_out_of_place is not similarity_for_out_of_place
    torch.testing.assert_close(affinity_inplace, affinity_out_of_place)


def test_do_softmax_top_k_tie_break_selects_first_matching_index() -> None:
    """Tied similarity values resolve via torch.topk's current lowest-index-first tie-break."""
    similarity = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])

    affinity = do_softmax(similarity, top_k=1)

    assert torch.equal(affinity, torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]))


def test_do_softmax_dense_branch_large_magnitude_input_is_finite_and_normalized() -> None:
    """Large-magnitude similarity values stay finite and normalize to 1 via the max-subtraction trick."""
    similarity = torch.tensor([[[0.0, 1e4], [1e4, 0.0], [-1e4, -1e4]]])

    affinity = do_softmax(similarity, top_k=None)

    assert torch.isfinite(affinity).all()
    torch.testing.assert_close(affinity.sum(dim=1), torch.ones(1, 2))


def _reference_readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    """Independent, elementwise reimplementation of readout's flatten+bmm+reshape composition."""
    b_dim, cv_dim, t_dim, h_dim, w_dim = mv.shape
    hw_dim = affinity.shape[2]
    out = torch.zeros(b_dim, cv_dim, hw_dim, dtype=mv.dtype)
    for b in range(b_dim):
        for cv in range(cv_dim):
            for hw in range(hw_dim):
                acc = 0.0
                n = 0
                for t in range(t_dim):
                    for h in range(h_dim):
                        for w in range(w_dim):
                            acc += mv[b, cv, t, h, w].item() * affinity[b, n, hw].item()
                            n += 1
                out[b, cv, hw] = acc
    return out.view(b_dim, cv_dim, h_dim, w_dim)


@pytest.mark.parametrize(
    'num_frames',
    [
        pytest.param(1, id='single-memory-frame'),
        pytest.param(3, id='multiple-memory-frames'),
    ],
)
def test_readout_matches_independent_reference_across_frame_counts(num_frames: int) -> None:
    """readout's batched matmul matches an elementwise reference for T=1 and T>1 memory frames."""
    height, width, channels = 1, 2, 2
    memory_values = torch.randn(1, channels, num_frames, height, width)
    affinity = torch.softmax(torch.randn(1, num_frames * height * width, height * width), dim=1)

    mem = readout(affinity, memory_values)

    expected = _reference_readout(affinity, memory_values)
    torch.testing.assert_close(mem, expected)


def test_get_affinity_equals_do_softmax_of_get_similarity() -> None:
    """get_affinity's composition matches calling do_softmax directly on get_similarity's output."""
    memory_key = torch.randn(1, 2, 2)
    memory_shrinkage = torch.rand(1, 1, 2) + 0.5
    query_key = torch.randn(1, 2, 2)
    query_selection = torch.rand(1, 2, 2) + 0.1

    affinity_via_shorthand = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
    affinity_via_composed_calls = do_softmax(get_similarity(memory_key, memory_shrinkage, query_key, query_selection))

    torch.testing.assert_close(affinity_via_shorthand, affinity_via_composed_calls)


@pytest.mark.parametrize(
    'has_qe',
    [
        pytest.param(True, id='qe-present'),
        pytest.param(False, id='qe-none'),
    ],
)
def test_get_similarity_backward_produces_finite_gradients(has_qe: bool) -> None:
    """Backprop through get_similarity yields finite gradients for mk, ms, qk (and qe when present)."""
    mk = torch.randn(1, 2, 2, requires_grad=True)
    ms = torch.randn(1, 1, 2, requires_grad=True)
    qk = torch.randn(1, 2, 2, requires_grad=True)
    qe = torch.randn(1, 2, 2, requires_grad=True) if has_qe else None

    similarity = get_similarity(mk, ms, qk, qe)
    similarity.sum().backward()

    assert mk.grad is not None
    assert torch.isfinite(mk.grad).all()
    assert ms.grad is not None
    assert torch.isfinite(ms.grad).all()
    assert qk.grad is not None
    assert torch.isfinite(qk.grad).all()
    if has_qe:
        assert qe.grad is not None
        assert torch.isfinite(qe.grad).all()


def test_get_affinity_backward_produces_finite_gradients() -> None:
    """Backprop through get_affinity's dense-softmax path yields finite gradients for all inputs."""
    mk = torch.randn(1, 2, 2, requires_grad=True)
    ms = torch.randn(1, 1, 2, requires_grad=True)
    qk = torch.randn(1, 2, 2, requires_grad=True)
    qe = torch.randn(1, 2, 2, requires_grad=True)

    affinity = get_affinity(mk, ms, qk, qe)
    affinity.sum().backward()

    assert mk.grad is not None
    assert torch.isfinite(mk.grad).all()
    assert ms.grad is not None
    assert torch.isfinite(ms.grad).all()
    assert qk.grad is not None
    assert torch.isfinite(qk.grad).all()
    assert qe.grad is not None
    assert torch.isfinite(qe.grad).all()


@pytest.mark.parametrize(
    'inplace',
    [
        pytest.param(True, id='inplace-true'),
        pytest.param(False, id='inplace-false'),
    ],
)
def test_do_softmax_top_k_branch_backward_raises_inplace_version_error(inplace: bool) -> None:
    """Pins current behavior: values.exp_() in the top-k path breaks backward regardless of inplace."""
    mk = torch.randn(1, 2, 2, requires_grad=True)
    ms = torch.randn(1, 1, 2, requires_grad=True)
    qk = torch.randn(1, 2, 2, requires_grad=True)
    qe = torch.randn(1, 2, 2, requires_grad=True)
    similarity = get_similarity(mk, ms, qk, qe)

    affinity = do_softmax(similarity, top_k=1, inplace=inplace)

    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        affinity.sum().backward()

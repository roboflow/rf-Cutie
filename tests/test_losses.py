"""Tests for temporal segmentation loss reductions."""

import torch

from cutie.model.losses import ce_loss, dice_loss


def test_ce_loss_matches_manual_soft_target_temporal_reduction() -> None:
    """Cross-entropy sums frame losses before averaging independently labelled points."""
    logits = torch.tensor(
        [
            [[3.0, -1.0], [-3.0, 1.0]],
            [[1.0, 2.0], [-1.0, -2.0]],
        ]
    )
    soft_targets = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )

    actual = ce_loss(logits, soft_targets)
    per_frame_point_loss = -(soft_targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
    expected = per_frame_point_loss.sum(dim=0).mean()

    torch.testing.assert_close(actual, expected)


def test_dice_loss_ignores_background_channel() -> None:
    """A perfect foreground mask has zero Dice loss despite incorrect background probabilities."""
    masks = torch.tensor(
        [
            [[0.9, 0.8], [0.0, 1.0]],
            [[0.1, 0.2], [1.0, 0.0]],
        ]
    )
    soft_targets = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    actual = dice_loss(masks, soft_targets)

    torch.testing.assert_close(actual, torch.tensor(0.0))


def test_dice_loss_uses_smoothed_foreground_overlap() -> None:
    """Imperfect foreground overlap follows the documented smoothed Dice formula."""
    masks = torch.tensor([[[0.0, 0.0], [0.25, 0.75]], [[0.0, 0.0], [1.0, 0.0]]])
    soft_targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])

    actual = dice_loss(masks, soft_targets)
    expected = torch.tensor(1.0 / 6.0)

    torch.testing.assert_close(actual, expected)

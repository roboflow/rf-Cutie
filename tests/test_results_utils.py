"""Optional evaluation dependency contracts for result saving."""

import importlib.util

import pytest


PIL_AVAILABLE = importlib.util.find_spec('PIL') is not None
pytestmark = pytest.mark.skipif(
    not PIL_AVAILABLE,
    reason='ResultSaver tests require the inference extra (Pillow).',
)

if PIL_AVAILABLE:
    from cutie.inference.utils import results_utils


def test_score_saving_requires_evaluation_extra(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Fail before starting a saver thread when hickle is unavailable."""
    monkeypatch.setattr(results_utils, 'hkl', None, raising=False)

    with pytest.raises(ModuleNotFoundError, match=r'cutie\[evaluation\]'):
        results_utils.ResultSaver(
            tmp_path,
            'video',
            dataset='generic',
            object_manager=None,
            use_long_id=False,
            save_scores=True,
        )


def test_burst_saving_requires_evaluation_extra(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Reject BURST output before starting a saver thread without pycocotools."""
    monkeypatch.setattr(results_utils, 'mask_util', None)

    with pytest.raises(ModuleNotFoundError, match=r'cutie\[evaluation\]'):
        results_utils.ResultSaver(
            tmp_path,
            'video',
            dataset='burst',
            object_manager=None,
            use_long_id=False,
            init_json={'segmentations': [], 'annotated_image_paths': []},
        )

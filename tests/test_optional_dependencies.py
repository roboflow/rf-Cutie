"""Import checks for installed optional dependency profiles."""

import importlib

import pytest


OPTIONAL_MODULES = (
    'PIL',
    'PySide6',
    'av',
    'cv2',
    'easydict',
    'git',
    'hickle',
    'hydra',
    'pycocotools.mask',
    'qdarktheme',
    'requests',
    'scipy',
    'thin_plate_spline',
    'tensorboard',
    'torchvision',
    'tqdm',
)


@pytest.mark.parametrize('module', OPTIONAL_MODULES)
def test_installed_optional_dependency_imports(module: str) -> None:
    """Import every optional package required by project extras."""
    importlib.import_module(module)

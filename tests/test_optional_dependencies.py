"""Import checks for installed optional dependency profiles."""

import importlib
import importlib.util

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
    'tensorboard',
    'tps',
    'torchvision',
    'tqdm',
)


def optional_module_is_available(module: str) -> bool:
    """Return whether an optional module can be located without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def optional_module_case(module: str):
    """Mark an optional module test skipped when its extra is not installed."""
    return pytest.param(
        module,
        marks=pytest.mark.skipif(
            not optional_module_is_available(module),
            reason=f'{module} is supplied by an optional dependency profile.',
        ),
    )


@pytest.mark.parametrize('module', [optional_module_case(module) for module in OPTIONAL_MODULES])
def test_installed_optional_dependency_imports(module: str) -> None:
    """Import every available optional package without requiring extras for core tests."""
    importlib.import_module(module)

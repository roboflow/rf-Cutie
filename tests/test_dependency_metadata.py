"""Contracts for the minimal runtime and optional dependency profiles."""

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PYPROJECT = ROOT / 'pyproject.toml'
EXTRA_IMPORTS = {
    'av': 'import av',
    'easydict': 'from easydict',
    'gitpython': 'import git',
    'hickle': 'import hickle',
    'hydra-core': 'import hydra',
    'opencv-python': 'import cv2',
    'pillow': 'from PIL',
    'pycocotools': 'import pycocotools',
    'pyqtdarktheme': 'import qdarktheme',
    'pyside6': 'from PySide6',
    'requests': 'import requests',
    'scipy': 'from scipy',
    'tensorboard': 'from torch.utils.tensorboard',
    'thinplate': 'import thinplate',
    'torchvision': 'from torchvision',
    'tqdm': 'from tqdm',
}


def requirement_name(requirement: str) -> str:
    """Return the normalized distribution name from a requirement string."""
    return re.match(r'[A-Za-z0-9_.-]+', requirement).group(0).lower()


def read_pyproject() -> dict:
    """Load the project metadata with the Python 3.10-compatible TOML parser."""
    with PYPROJECT.open('rb') as file:
        return tomllib.load(file)


def test_base_dependencies_are_only_model_core() -> None:
    """Keep a base installation limited to model construction dependencies."""
    metadata = read_pyproject()

    assert {requirement_name(item) for item in metadata['project']['dependencies']} == {
        'einops',
        'numpy',
        'omegaconf',
        'torch',
    }


def test_every_optional_dependency_has_a_local_import() -> None:
    """Require each published extra to be justified by a direct or guarded import."""
    metadata = read_pyproject()
    optional_dependencies = metadata['project']['optional-dependencies']
    declared = {
        requirement_name(requirement)
        for requirements in optional_dependencies.values()
        for requirement in requirements
    }
    source = '\n'.join(
        path.read_text()
        for directory in ('cutie', 'gui', 'scripts', 'examples')
        for path in (ROOT / directory).rglob('*.py')
    )

    assert declared == set(EXTRA_IMPORTS)
    for package, import_statement in EXTRA_IMPORTS.items():
        assert import_statement in source, package


def test_cython_is_build_only() -> None:
    """Keep the un-packaged RITM extension compiler out of runtime extras."""
    metadata = read_pyproject()

    assert metadata['dependency-groups']['build'] == ['cython']
    assert 'cimport cython' in (ROOT / 'gui/ritm/utils/cython/_get_dist_maps.pyx').read_text()


def test_test_group_contains_only_test_tooling() -> None:
    """Avoid duplicating runtime dependencies in the test-only group."""
    metadata = read_pyproject()

    assert {requirement_name(item) for item in metadata['dependency-groups']['tests']} == {
        'coverage',
        'pytest',
        'tomli',
    }

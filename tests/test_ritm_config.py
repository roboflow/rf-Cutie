"""Tests for the RITM configuration mapping compatibility layer."""

import torch

from gui.ritm.utils.exp import load_config_file


def test_load_config_file_preserves_mapping_and_attribute_access(tmp_path) -> None:
    """RITM configs support nested access and arbitrary runtime values."""
    config_path = tmp_path / 'model.yml'
    config_path.write_text('model:\n  name: cutie\n  sizes: [1, 2]\n', encoding='utf-8')

    config = load_config_file(config_path, return_edict=True)
    config.device = torch.device('cpu')
    config['runtime'] = {'enabled': True}

    assert config.model.name == 'cutie'
    assert config['model']['sizes'] == [1, 2]
    assert config.device == torch.device('cpu')
    assert config.runtime.enabled is True

import logging
import os
import sys

# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    # Prefer workspace images, then --images, then --video. Existing workspace masks initialize
    # the session so an interrupted annotation run can continue with the same workspace.
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument(
        '--workspace',
        help='directory for storing buffered images (if needed) and output masks',
        default=None,
    )
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument(
        '--workspace_init_only', action='store_true', help='initialize the workspace and exit'
    )

    args = parser.parse_args()
    return args


if __name__ in '__main__':
    # input arguments
    args = get_arguments()

    # perform slow imports after parsing args
    import torch
    from omegaconf import open_dict
    from hydra import compose, initialize
    from PySide6.QtWidgets import QApplication
    import qdarktheme
    from gui.main_controller import MainController

    # logging
    log = logging.getLogger()

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path='../cutie/config', job_name='gui')
    cfg = compose(config_name='gui_config')

    # general setup
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    args.device = device
    log.info(f'Using device: {device}')

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            assert k not in cfg, f'Argument {k} already exists in config'
            cfg[k] = v

    # start everything
    app = QApplication(sys.argv)
    qdarktheme.setup_theme('auto')
    ex = MainController(cfg)
    if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
        sys.exit(0)
    else:
        sys.exit(app.exec())

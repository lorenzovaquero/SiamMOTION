"""log_utils.py: Utilities for logging"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import logging.handlers
from .ModuleNameFilter import ModuleNameFilter

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"

LOG_FORMAT_DEFAULT = '%(asctime)s [%(funcName)s] %(levelname)s: %(message)s'
LOG_FORMAT_PRINT = '%(message)s'

ALLOWED_MODULES = ['root', 'scripts', 'conversion', 'display', 'inference', 'misc', 'neuralNetwork', 'tracking', 'training', 'evaluation', 'parameters', 'logger', 'SiamMT']


def setup_log(logger, allowed_modules=ALLOWED_MODULES, level='info', output='stdout', format_style=LOG_FORMAT_DEFAULT):
    log_formatter = logging.Formatter(format_style)

    if level.lower() == 'CRITICAL'.lower():
        log_level = logging.CRITICAL

    elif level.lower() == 'ERROR'.lower():
        log_level = logging.ERROR

    elif level.lower() == 'WARNING'.lower():
        log_level = logging.WARNING

    elif level.lower() == 'INFO'.lower():
        log_level = logging.INFO

    elif level.lower() == 'DEBUG'.lower():
        log_level = logging.DEBUG

    else:
        print('error: unknown log level %s' % level)
        sys.exit(1)

    if output.lower() == 'stdout'.lower() or output.lower() == 'stream'.lower():
        log_handler = logging.StreamHandler(stream=sys.stdout)
        log_handler.name = 'stdout logger'
        log_handler.level = log_level
        log_handler.formatter = log_formatter

    elif output.lower() == 'stderr'.lower():
        log_handler = logging.StreamHandler(stream=sys.stderr)
        log_handler.name = 'stderr logger'
        log_handler.level = log_level
        log_handler.formatter = log_formatter

    elif output.lower() == 'syslog'.lower():
        log_handler = logging.handlers.SysLogHandler(address='/dev/log')
        log_handler.name = 'syslog logger'
        log_handler.level = log_level
        log_handler.formatter = log_formatter

    else:
        print('error: unknown log output %s' % output)
        sys.exit(1)

    log_handler.addFilter(ModuleNameFilter(allowed_modules))

    # Create logger
    logger.setLevel(log_level)
    logger.addHandler(log_handler)

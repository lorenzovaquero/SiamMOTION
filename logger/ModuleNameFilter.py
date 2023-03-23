"""ModuleNameFilter.py: Log filter for printing only the desired modules"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class ModuleNameFilter(logging.Filter):
    """Filters the log, in order to only print the desired modules"""

    def __init__(self, allowed_modules):
        logging.Filter.__init__(self)
        self.allowed_modules = set(allowed_modules)

    def filter(self, record):
        return record.name.split('.')[0] in self.allowed_modules

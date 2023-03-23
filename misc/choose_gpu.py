"""autoselect_gpu.py: Utility for selecting the best GPU on the machine"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess as sp
import os
import logging

logger = logging.getLogger(__name__)

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"

def choose_gpu(gpu_id=-1, acceptable_available_memory=2048):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if gpu_id == -1:
        gpu_list = get_unused_gpus(acceptable_available_memory)

        if len(gpu_list) < 1:
            raise OSError('There aren\'t any available GPUs with enough memory')
        else:
            gpu_id = gpu_list[0][0]

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    return gpu_id



def get_unused_gpus(acceptable_available_memory=2048):
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        available_gpus = [(i, x) for i, x in enumerate(memory_free_values) if x > acceptable_available_memory]
        available_gpus = sorted(available_gpus, key=lambda x: x[1], reverse=True)

        return available_gpus

    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs couldn\'t be listed.', e)
        return 0

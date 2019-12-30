import numpy as np
import os

def get_free_gpu():
    """
    Get the index of the gpus with the most available GPU memory

    Inputs:
    - None

    Returns:
    - idx: an integer scalar which is the index of the returned GPU 
    """
    tmp_file = 'tmp_gpu'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + tmp_file)
    memory_available = [int(x.split()[2]) for x in open('tmp_gpu', 'r').readlines()]
    os.system('rm '+tmp_file)
    if len(memory_available) == 0:
      return None # no gpu available
    idx =  np.argmax(memory_available)
    return idx

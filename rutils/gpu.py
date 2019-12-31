import numpy as np
import os
import time
import random

def get_free_gpu():
    """
    Get the index of the gpus with the most available GPU memory

    Inputs:
    - None

    Returns:
    - idx: an integer scalar which is the index of the returned GPU 
    """
    time.sleep(random.random()*2) 
    tmp_file = 'tmp_gpu'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + tmp_file)
    memory_available = [int(x.split()[2]) for x in open(tmp_file, 'r').readlines()]
    os.system('rm '+tmp_file)

    os.system('nvidia-smi -q|grep Processes>'+tmp_file)
    proc_available = [len(x.split())>=3 for x in open(tmp_file, 'r').readlines()]
    os.system('rm '+tmp_file)


    if len(memory_available) == 0:
      return None # no gpu available

    stat_list = [(proc_available[idx], memory_available[idx], idx) for idx in range(len(proc_available))]
    #idx =  np.argmax(memory_available)
    idx = sorted(stat_list, reverse=True)[0][2]
    return int(idx)

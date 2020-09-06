import numpy as np
import os
import time
import random
from filelock import FileLock
from .time import get_current_time
import socket

def get_free_gpus(hostname=None):
    if hostname is None:
        hostname = socket.gethostname()

    lock = FileLock(hostname + '_get_free_gpu.lock')
    with lock:
      tmp_file = hostname + '_tmp_gpu'

      os.system('nvidia-smi -q|grep Processes>'+tmp_file)
      proc_available = [len(x.split())>=3 for x in open(tmp_file, 'r').readlines()]
      os.system('rm '+tmp_file)

    free_gpus = []
    for idx in range(len(proc_available)):
        if proc_available[idx] is True:
            free_gpus.append(idx)

    return free_gpus

def get_free_gpus_from_hostlist(hostlist=None):
    if hostlist is None:
        hostlist = ["gpu-cn{:03d}".format(x) for x in range(1, 18)]

    free_gpus_dict = {}
    for hostname in hostlist:
        tmp_file = hostname + '_tmp_gpu'
        os.system('ssh '+ hostname + ' nvidia-smi -q|grep Processes>'+tmp_file)
        proc_available = [len(x.split())>=3 for x in open(tmp_file, 'r').readlines()]
        os.system('rm '+tmp_file)

        free_gpus = []
        for idx in range(len(proc_available)):
            if proc_available[idx] is True:
                free_gpus.append(idx)
        if len(free_gpus) > 0:
            free_gpus_dict[hostname] = free_gpus

    return free_gpus_dict

def get_free_gpu():
    """
    Get the index of the gpus with the most available GPU memory

    Inputs:
    - None

    Returns:
    - idx: an integer scalar which is the index of the returned GPU 
    """
    hostname = socket.gethostname()
    lock = FileLock(hostname + '_get_free_gpu.lock')
    #time.sleep(random.random())
    with lock:
      tmp_file = hostname + '_tmp_gpu'
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

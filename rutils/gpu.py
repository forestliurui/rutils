import numpy as np

def get_free_gpu():
    """
    Get the index of the gpus with the most available GPU memory

    Inputs:
    - None

    Returns:
    - idx: an integer scalar which is the index of the returned GPU 
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_gpu')
    memory_available = [int(x.split()[2]) for x in open('tmp_gpu', 'r').readlines()]
    idx =  np.argmax(memory_available)
    return idx

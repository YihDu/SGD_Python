import numpy as np
import torch
import ot
import time
from concurrent.futures import ProcessPoolExecutor
from GK_emd import *
import dill as pickle



def kernel_task(task):
    s1, s2, kernel , sigma  = task
    return kernel(s1, s2 , sigma)

def disc(samples1 , samples2 , kernel , sigma):
    n = len(samples1)
    m = len(samples2)
    
    loop_start_time = time.time()
    
    tasks = [(s1 , s2 , kernel, sigma) for s1 in samples1 for s2 in samples2]
    
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(kernel_task , tasks))
    
    d = sum(results) / (n * m)
    
    loop_end_time = time.time()
    print(f"Entire Loop took {loop_end_time - loop_start_time:.5f} seconds.") 
    
    return d


def compute_mmd(samples1, samples2, kernel, sigma ,is_hist=True):
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    return disc(samples1, samples1, kernel , sigma) + \
            disc(samples2, samples2, kernel , sigma) - \
            2 * disc(samples1, samples2, kernel , sigma)
            


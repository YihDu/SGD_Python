import numpy as np
import torch
import ot
import time
from concurrent.futures import ProcessPoolExecutor
from GK_emd import *
import dill as pickle
from tqdm import tqdm

def kernel_task(task):
    s1, s2, kernel , sigma  = task
    start_time = time.time()
    result = kernel(s1, s2, sigma)
    
    end_time = time.time()
    
    print(f"Task took {end_time - start_time:.5f} seconds.")
    return result.item()

def disc(samples1 , samples2 , kernel , sigma , max_workers=3):
    n = len(samples1)
    m = len(samples2)
    loop_start_time = time.time()
    
    tasks = [(s1 , s2 , kernel, sigma) for s1 in samples1 for s2 in samples2]
   
    results = []
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(kernel_task, tasks), total=len(tasks)):
                results.append(result)
    else:
        for task in tqdm(tasks, total=len(tasks)):
            results.append(kernel_task(task))
    
    
    # 打印出result的每一个元素
    print(results)
    
    d = sum(results) / (n * m)
    
    loop_end_time = time.time()
    print(f"Entire Loop took {loop_end_time - loop_start_time:.5f} seconds.") 
    
    return d
'''
def kernel_task(s1, s2, kernel, sigma):
    return kernel(s1, s2, sigma).item()

def disc(samples1, samples2, kernel, sigma):
    n = len(samples1)
    m = len(samples2)
    loop_start_time = time.time()
    
    results = []
    for s1 in samples1:
        for s2 in samples2:
            # 开始计算第i次
            time_start = time.time()
            result = kernel_task(s1, s2, kernel, sigma)
            
            # 结束计算第i次
            time_end = time.time()
            
            print(f"Time taken for one iteration: {time_end - time_start:.5f} seconds.")
            
            results.append(result)
    
    # 打印出result的每一个元素
    print(results)
    
    d = sum(results) / (n * m)
    
    loop_end_time = time.time()
    print(f"Entire Loop took {loop_end_time - loop_start_time:.5f} seconds.") 
    
    return d
'''

def compute_mmd(samples1, samples2, kernel, sigma ,is_hist=True):
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    return disc(samples1, samples1, kernel , sigma) + \
            disc(samples2, samples2, kernel , sigma) - \
            2 * disc(samples1, samples2, kernel , sigma)
            

if __name__ == "__main__":
    # test data
    array1 = np.random.rand(10, 11000, 7)
    array2 = np.random.rand(10, 11000, 7)
    
    sigma = 1e-4
    
    # 计算MMD
    mmd1 = compute_mmd(array1, array2, gaussian_emd, sigma)
    # mmd2 = compute_mmd(array1, array2, gaussian_emd_Sinkhorn, sigma)
    print(f"MMD using kernel1: {mmd1}")
    # print(f"MMD using kernel2: {mmd2}")
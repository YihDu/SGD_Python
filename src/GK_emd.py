import numpy as np
import ot
import torch
import torch.nn as nn
import sinkhorn_pointcloud as spc
import time
## POT - Wasserstein Distance

def gaussian_emd(x, y ,sigma):
    n = len(x)
    m = len(y)
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m

    cost_matrix = ot.dist(x, y, metric='euclidean')
    emd = ot.emd2(a, b, cost_matrix , numItermax=1000000000000)
    
    return np.exp(- emd * emd / (2 * sigma * sigma))

def gaussian_euclidean(x, y ,sigma):
    n = len(x)
    m = len(y)
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m
    euclidean_distance = np.linalg.norm(x - y)
    
    return np.exp(- euclidean_distance * euclidean_distance / (2 * sigma * sigma))

## PyTorch Sinkhorn to calculate Wasserstein distance

def gaussian_emd_Sinkhorn(x , y , sigma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.FloatTensor(x).to(device)
    y = torch.FloatTensor(y).to(device)
    
    n = len(x)
    
    # Sinkhorn parameters
    epsilon = 1e-2
    niter = 10000000000
    
    l1 = spc.sinkhorn_loss(x,y,epsilon,n,niter ,device)
    
    distance = l1
    
    result = np.exp(- distance.item() * distance.item() / (2 * sigma * sigma))
    
    return result


if __name__ == "__main__":
    # for test
    np.random.seed(0)
    x = np.random.rand(1000 , 4)
    y = np.random.rand(1000 , 4)
    
    print(x)
    print(y)
    
    sigma = 1.0

    time_start = time.time()
    
    
    result_classical = gaussian_emd(x, y, sigma)
    
    time1 = time.time()
    
    print(f"Time taken for classical: {time1 - time_start:.5f} seconds.")
    print("Gaussian EMD (classical):", result_classical)


    # result_sinkhorn = gaussian_emd_Sinkhorn(x, y, sigma)
    
    # time2 = time.time()
    # print(f"Time taken for Sinkhorn: {time2 - time1:.5f} seconds.")
    # print("Gaussian EMD (Sinkhorn):", result_sinkhorn)
    
    
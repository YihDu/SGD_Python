import numpy as np
import ot
import torch
import torch.nn as nn
import sinkhorn_pointcloud as spc
## POT库实现Wasserstein距离

def gaussian_emd(x, y ,sigma):
    n = len(x)
    m = len(y)
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m

    cost_matrix = ot.dist(x, y, metric='euclidean')
    emd = ot.emd2(a, b, cost_matrix , numItermax=10000)

    return np.exp(- emd * emd / (2 * sigma * sigma))


## PyTorch Sinkhorn迭代算法 实现Wasserstein距离

def gaussian_emd_Sinkhorn(x , y , sigma):


    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    n = len(x)
    
    # Sinkhorn parameters
    epsilon = 0.01
    niter = 10000
    
    l1 = spc.sinkhorn_loss(x,y,epsilon,n,niter)
    
    distance = l1

    return np.exp(- distance * distance / (2 * sigma * sigma))


if __name__ == "__main__":
    # 生成示例数据
    x = np.random.rand(100, 3)
    y = np.random.rand(100, 3)
    
    print(x)
    print(y)
    
    sigma = 1.0

    result_classical = gaussian_emd(x, y, sigma)
    print("Gaussian EMD (classical):", result_classical)

    result_sinkhorn = gaussian_emd_Sinkhorn(x, y, sigma)
    print("Gaussian EMD (Sinkhorn):", result_sinkhorn)
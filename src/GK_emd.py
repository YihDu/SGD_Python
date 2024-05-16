import numpy as np
import ot
import json


## POT库实现Wasserstein距离

def gaussian_emd(x, y ,sigma):
    X = np.stack(x)
    Y = np.stack(y)
    
    n = len(x)
    m = len(y)
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m

    cost_matrix = ot.dist(X, Y, metric='euclidean')
    emd = ot.emd2(a, b, cost_matrix , numItermax=100000000)

    return np.exp(-emd * emd / (2 * sigma * sigma))


## PyTorch Sinkhorn迭代算法 实现Wasserstein距离

import torch
from torch.autograd import Variable

def sinkhorn_normalized(x, y, epsilon, n, niter):

    Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
    return 2 * Wxy - Wxx - Wyy

def sinkhorn_loss(x, y, epsilon, n, niter):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    print("begin cost matrix")
    
    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(cost_matrix(x, y))  # Wasserstein cost function

    print("end cost matrix")
    
    print("begin marginals")
    
    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()
        
        print(i)
        print(err)
        
        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


def gaussian_emd_Sinkhorn(x , y , sigma):
    
    print("开始转换为Tensor")
    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    print("结束转换为Tensor")
    num_samples = len(x)
    
    l2 =  sinkhorn_normalized(x, y , epsilon = 0.1 , n = num_samples ,niter = 1000)
    emd = l2.item()
    
    return np.exp(-emd * emd / (2 * sigma * sigma))
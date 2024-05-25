import torch
from torch.autograd import Variable

def sinkhorn_loss(x, y, epsilon, n, niter , device):

    C = Variable(cost_matrix(x, y,device))  #


    mu = Variable(1. / n * torch.ones(n, device=device), requires_grad=False)
    nu = Variable(1. / n * torch.ones(n, device=device), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -0.8  # nesterov-like acceleration
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
        return torch.log(torch.exp(A).sum(dim=1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v = 0. * mu, 0. * nu
    

    for _ in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v

        if (u - u1).abs().sum() < thresh:
            break

    pi = torch.exp(M(u, v))  # Transport plan
    cost = torch.sum(pi * C)  # Sinkhorn cost
    return cost


def cost_matrix(x, y,device): 
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** 2, 2)
    return c.to(device)
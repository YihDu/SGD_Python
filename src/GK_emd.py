import numpy as np
import ot
import json




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
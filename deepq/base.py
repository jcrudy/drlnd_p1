from multipledispatch.dispatcher import Dispatcher
from torch.tensor import Tensor
from numpy import ndarray
import torch
import numpy as np

torchify = Dispatcher('torchify')

@torchify.register(Tensor)
def torchify_tensor(tens):
    return tens

@torchify.register(ndarray)
def torchify_numpy(arr):
    return torch.from_numpy(arr)

def torchify32(x):
    return torchify(numpify(x).astype(np.float32))

numpify = Dispatcher('numpify')

@numpify.register(Tensor)
def numpify_tensor(tens):
    return tens.detach().numpy()

@numpify.register(ndarray)
def numpify_numpy(arr):
    return arr

def rolling_mean(y, window):
    result = (
              np.convolve(y, np.ones(shape=window), mode='full') / 
              np.convolve(np.ones_like(y), np.ones(shape=window), mode='full')
              )
    return result[:(1-window)]
    

# def rolling_mean(x, y, window) :
#     result = np.cumsum(y, dtype=float)
#     result[window:] = result[window:] - result[:-window]
#     return x[window:-window], result[window - 1:] / window


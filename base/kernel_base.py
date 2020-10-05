import numpy as np
import scipy.io as scio
import math


def Gradient_Similarity(gradB, gradT):
    '''
    calculate the gradient similarity using the response of convolution devided by the norm
    Reference: Z. Hu and M.-H. Yang. Good regions to deblur. In ECCV 2020
    '''
    gradB = gradB.astype(np.float64)
    gradT = gradT.astype(np.float64)




    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

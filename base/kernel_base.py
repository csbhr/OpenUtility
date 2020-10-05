import numpy as np
import math
import cv2


def Gradient_Similarity(gradB, gradT):
    '''
    calculate the gradient similarity using the response of convolution devided by the norm
    Reference: Z. Hu and M.-H. Yang. Good regions to deblur. In ECCV 2012
    '''
    gradB = gradB.astype(np.float64)
    gradT = gradT.astype(np.float64)

    gradB_sum = math.sqrt(np.sum(np.power(gradB, 2)))
    gradT_sum = math.sqrt(np.sum(np.power(gradT, 2)))

    kb, kt = gradB.shape[0], gradT.shape[0]
    psize = kt // 2
    gradB_norm = gradB / gradB_sum
    gradB_pad = np.zeros(shape=[kb + 2 * psize, kb + 2 * psize], dtype=np.float64)
    gradB_pad[psize:-psize, psize:-psize] = gradB_norm
    corr = cv2.filter2D(src=gradB_pad, ddepth=-1, kernel=gradT, borderType=cv2.BORDER_CONSTANT)

    similarity = np.max(corr) / gradT_sum

    return similarity

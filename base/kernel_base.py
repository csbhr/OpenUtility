import numpy as np
import math
import cv2
from scipy import io as scio
from base.image_base import RMSE, PSNR, SSIM


def load_mat_kernel(mat_path):
    data_dict = scio.loadmat(mat_path)
    key = [v for v in data_dict.keys() if v not in ['__header__', '__version__', '__globals__']][0]
    return data_dict[key]


def kernel2png(kernel):
    # kernel: [ks, ks], float32/float64
    # return kernel_png: [ks, ks, 3], uint8
    kernel = cv2.resize(kernel, dsize=(0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    kernel = np.clip(kernel, 0, np.max(kernel))
    kernel = kernel / np.sum(kernel)
    mi = np.min(kernel)
    ma = np.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = np.round(np.clip(kernel * 255., 0, 255))
    kernel_png = np.stack([kernel, kernel, kernel], axis=2).astype('uint8')
    return kernel_png


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


def Kernel_RMSE_PSNR_SSIM(kernel1, kernel2):
    h1, w1 = kernel1.shape
    h2, w2 = kernel2.shape

    mh, mw = max(h1, h2), max(w1, w2)
    kernel1_pad = np.zeros(shape=(mh, mw)).astype(kernel1.dtype)
    kernel2_pad = np.zeros(shape=(mh, mw)).astype(kernel2.dtype)
    kernel1_pad[(mh - h1) // 2:(mh - h1) // 2 + h1, (mw - w1) // 2:(mw - w1) // 2 + w1] = kernel1
    kernel2_pad[(mh - h2) // 2:(mh - h2) // 2 + h2, (mw - w2) // 2:(mw - w2) // 2 + w2] = kernel2

    # kernel1_pad = cv2.resize(kernel1, dsize=(w2, w2), interpolation=cv2.INTER_CUBIC)
    # kernel2_pad = kernel2

    rmse = RMSE(kernel1_pad, kernel2_pad)
    psnr = PSNR(kernel1_pad * 255, kernel2_pad * 255)
    ssim = SSIM(kernel1_pad * 255, kernel2_pad * 255)
    return rmse, psnr, ssim

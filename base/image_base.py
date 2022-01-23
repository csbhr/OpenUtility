import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import cv2
from base.matlab_imresize import imresize


#################################################################################
####                            Image Operation                              ####
#################################################################################
def matlab_imresize(img, scalar_scale=None, output_shape=None, method='bicubic'):
    '''same as matlab2017 imresize
    img: shape=[h, w, c]
    scalar_scale: the resize scale
        if None, using output_shape
    output_shape: the resize shape, (h, w)
        if scalar_scale=None, using this param
    method: the interpolation method
        optional: 'bicubic', 'bilinear'
        default: 'bicubic'
    '''
    return imresize(
        I=img,
        scalar_scale=scalar_scale,
        output_shape=output_shape,
        method=method
    )


def image_shift(img, offset_x=0., offset_y=0.):
    '''shift the img by (offset_x, offset_y) on (axis-x, axis-y)
    img: numpy.array, shape=[h, w, c]
    offset_x: offset pixels on axis-x
        positive=left; negative=right
    offset_y: offset pixels on axis-y
        positive=up; negative=down
    '''
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    B, C, H, W = img_tensor.size()

    # init flow
    flo = torch.ones(B, 2, H, W).type_as(img_tensor)
    flo[:, 0, :, :] *= offset_x
    flo[:, 1, :, :] *= offset_y

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # Interpolation
    vgrid = vgrid.permute(0, 2, 3, 1)
    output_tensor = F.grid_sample(img_tensor, vgrid, padding_mode='border')

    output = output_tensor.round()[0].detach().numpy().transpose(1, 2, 0).astype(img.dtype)

    return output


def rgb2ycbcr(img, range=255., only_y=True):
    """same as matlab rgb2ycbcr, please use bgr2ycbcr when using cv2.imread
    img: shape=[h, w, 3]
    range: the data range
    only_y: only return Y channel
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    range_scale = 255. / range
    img *= range_scale

    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    rlt /= range_scale
    if in_img_type == np.uint8:
        rlt = rlt.round()
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, range=255., only_y=True):
    """bgr version of rgb2ycbcr, for cv2.imread
    img: shape=[h, w, 3]
    range: the data range
    only_y: only return Y channel
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    range_scale = 255. / range
    img *= range_scale

    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]

    rlt /= range_scale
    if in_img_type == np.uint8:
        rlt = rlt.round()
    return rlt.astype(in_img_type)


def ycbcr2rgb(img, range=255.):
    """same as matlab ycbcr2rgb
    img: shape=[h, w, 3]
    range: the data range
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    range_scale = 255. / range
    img *= range_scale

    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    rlt /= range_scale
    if in_img_type == np.uint8:
        rlt = rlt.round()
    return rlt.astype(in_img_type)


#################################################################################
####                            Image PSNR SSIM                              ####
#################################################################################

def RMSE(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return math.sqrt(mse)


def PSNR(img1, img2):
    '''
    img1, img2: [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def SSIM(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def PSNR_SSIM_Shift_Best(img1, img2, window_size=5):
    '''
    img1, img2: [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1 = img1[window_size:-window_size, window_size:-window_size, :]
    best_psnr, best_ssim = 0., 0.
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            shifted_img2 = image_shift(img2, offset_x=i, offset_y=j)
            shifted_img2 = shifted_img2[window_size:-window_size, window_size:-window_size, :]
            psnr = PSNR(img1, shifted_img2)
            ssim = SSIM(img1, shifted_img2)
            best_psnr = max(best_psnr, psnr)
            best_ssim = max(best_ssim, ssim)
    return best_psnr, best_ssim


#################################################################################
####                                Others                                   ####
#################################################################################


def evaluate_smooth(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    smooth = np.mean(dst)
    return smooth


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img


def warp_by_flow(x, flo, device='cuda'):
    B, C, H, W = flo.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.to(device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode='border')

    return output

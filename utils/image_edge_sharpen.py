import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
from utils.video_metric_utils import batch_calc_video_PSNR_SSIM


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("mkdir:", dir)


def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_blur_kernel():
    gaussian_sigma = 1.0
    gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
    kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
    return kernel


def get_blur(img, kernel):
    img = np.array(img).astype('float32')
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    img_tensor = F.pad(img_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(img_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

    blur_img = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('float32')

    return blur_img


def image_post_process_results(ori_root, save_root, alpha=3.):
    handle_dir(save_root)

    guassian_kernel = get_blur_kernel()

    frame_names = sorted(os.listdir(os.path.join(ori_root)))
    for fn in frame_names:
        ori_img = cv2.imread(os.path.join(ori_root, fn)).astype('float32')
        blur_img = get_blur(ori_img, guassian_kernel).astype('float32')

        res_img = ori_img - blur_img

        result = blur_img + alpha * res_img

        basename = fn.split(".")[0]
        cv2.imwrite(os.path.join(save_root, "{}_post.png".format(basename)), result)

        ##
        # res_img = np.clip(res_img, 0, np.max(res_img))
        # res_img = (res_img / np.max(res_img)) * 255.
        # cv2.imwrite(os.path.join(save_root, "{}_res.png".format(basename)), res_img)
        ##

        print("{} done!".format(fn))


def video_post_process_results(ori_root, save_root, alpha=3.):
    handle_dir(save_root)

    guassian_kernel = get_blur_kernel()

    video_names = sorted(os.listdir(ori_root))
    for vn in video_names:
        handle_dir(os.path.join(save_root, vn))

        frame_names = sorted(os.listdir(os.path.join(ori_root, vn)))
        for fn in frame_names:
            ori_img = cv2.imread(os.path.join(ori_root, vn, fn)).astype('float32')
            blur_img = get_blur(ori_img, guassian_kernel).astype('float32')

            res_img = ori_img - blur_img

            result = blur_img + alpha * res_img

            basename = fn.split(".")[0]
            cv2.imwrite(os.path.join(save_root, vn, "{}_post.png".format(basename)), result)

            ##
            # res_img = np.clip(res_img, 0, np.max(res_img))
            # res_img = (res_img / np.max(res_img)) * 255.
            # cv2.imwrite(os.path.join(save_root, vn, "{}_res.png".format(basename)), res_img)
            ##

            print("{}-{} done!".format(vn, fn))


if __name__ == '__main__':
    # image_post_process_results(
    #     ori_root='/media/csbhr/Bear/Dataset/FaceSR/face/test/bic',
    #     save_root='./temp/edge/residual',
    #     alpha=2
    # )

    root_list = []
    for i in range(21):
        alpha = 1.0 + i * 0.1
        save_root = '/home/csbhr/Disk-2T/work/OpenUtility/temp/edge_sharpen/post_{}'.format(alpha)
        video_post_process_results(
            ori_root='/home/csbhr/Disk-2T/work/OpenUtility/temp/edge_sharpen/ori',
            save_root=save_root,
            alpha=alpha
        )
        root_list.append({
            'output': save_root,
            'gt': '/home/csbhr/Disk-2T/work/OpenUtility/temp/edge_sharpen/HR'
        })
    batch_calc_video_PSNR_SSIM(root_list)

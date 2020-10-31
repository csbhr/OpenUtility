import os
import cv2
import torch
import numpy as np
from base import image_base
from base.os_base import listdir
import utils.PerceptualSimilarity.models as lpips_models


def calc_image_PSNR_SSIM(output_root, gt_root, crop_border=4, shift_window_size=0, test_ycbcr=False, crop_GT=False):
    '''
    计算图片的 PSNR、SSIM，使用 EDVR 的计算方式
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    if test_ycbcr:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    PSNR_list = []
    SSIM_list = []
    output_img_list = sorted(listdir(output_root))
    gt_img_list = sorted(listdir(gt_root))
    for o_im, g_im in zip(output_img_list, gt_img_list):
        o_im_path = os.path.join(output_root, o_im)
        g_im_path = os.path.join(gt_root, g_im)
        im_GT = cv2.imread(g_im_path) / 255.
        im_Gen = cv2.imread(o_im_path) / 255.

        if crop_GT:
            h, w, c = im_Gen.shape
            im_GT = im_GT[:h, :w, :]  # crop GT to output size

        if test_ycbcr and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT = image_base.bgr2ycbcr(im_GT, range=1.)
            im_Gen = image_base.bgr2ycbcr(im_Gen, range=1.)

        # crop borders
        if crop_border != 0:
            if im_GT.ndim == 3:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT.ndim == 2:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT.ndim))
        else:
            cropped_GT = im_GT
            cropped_Gen = im_Gen

        if shift_window_size == 0:
            psnr = image_base.PSNR(cropped_GT * 255, cropped_Gen * 255)
            ssim = image_base.SSIM(cropped_GT * 255, cropped_Gen * 255)
        else:
            psnr, ssim = image_base.PSNR_SSIM_Shift_Best(cropped_GT * 255, cropped_Gen * 255,
                                                         window_size=shift_window_size)
        PSNR_list.append(psnr)
        SSIM_list.append(ssim)

        print("{} PSNR={:.5}, SSIM={:.4}".format(o_im, psnr, ssim))

    log = 'Average PSNR={:.5}, SSIM={:.4}'.format(sum(PSNR_list) / len(PSNR_list), sum(SSIM_list) / len(SSIM_list))
    print(log)

    return PSNR_list, SSIM_list, log


def batch_calc_image_PSNR_SSIM(root_list, crop_border=4, shift_window_size=0, test_ycbcr=False, crop_GT=False):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output images
            gt: the dir of gt images
    optional params:
        crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
        shift_window_size: defalut=0, if >0, shifting image within a window for best metric
        test_ycbcr: default=False, if True, applying Ycbcr color space
        crop_GT: default=False, if True, cropping GT to output size
    return:
        log_list: a list, each item is a dictionary that given two key-values:
            data_path: the evaluated dir
            log: the log of this dir
    '''
    log_list = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  Now Evaluation >>>>")
        print(">>>>  OUTPUT: {}".format(ouput_root))
        print(">>>>  GT: {}".format(gt_root))
        _, _, log = calc_image_PSNR_SSIM(
            ouput_root, gt_root, crop_border=crop_border, shift_window_size=shift_window_size,
            test_ycbcr=test_ycbcr, crop_GT=crop_GT
        )
        log_list.append({
            'data_path': ouput_root,
            'log': log
        })

    print("--------------------------------------------------------------------------------------")
    for i, log in enumerate(log_list):
        print("## The {}-th:".format(i))
        print(">> ", log['data_path'])
        print(">> ", log['log'])

    return log_list


def calc_image_LPIPS(output_root, gt_root, model=None, use_gpu=False, spatial=True):
    '''
    计算图片的 LPIPS
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    def _load_image(path, size=(512, 512)):
        img = cv2.imread(path)
        h, w, c = img.shape
        if h != size[0] or w != size[1]:
            img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
        return img[:, :, ::-1]

    def _im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):
        return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

    if model is None:
        model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)

    LPIPS_list = []
    output_img_list = sorted(listdir(output_root))
    gt_img_list = sorted(listdir(gt_root))
    for o_im, g_im in zip(output_img_list, gt_img_list):
        o_im_path = os.path.join(output_root, o_im)
        g_im_path = os.path.join(gt_root, g_im)
        im_GT = _im2tensor(_load_image(g_im_path))
        im_Gen = _im2tensor(_load_image(o_im_path))

        if use_gpu:
            im_GT = im_GT.cuda()
            im_Gen = im_Gen.cuda()

        lpips = model.forward(im_GT, im_Gen).mean()
        LPIPS_list.append(lpips)

        print("{} LPIPS={:.4}".format(o_im, lpips))

    log = 'Average LPIPS={:.4}'.format(sum(LPIPS_list) / len(LPIPS_list))
    print(log)

    return LPIPS_list, log


def batch_calc_image_LPIPS(root_list, use_gpu=False, spatial=True):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output images
            gt: the dir of gt images
    optional params:
        use_gpu: defalut=False, if True, using gpu
        spatial: default=True, if True, return spatial map
    return:
        log_list: a list, each item is a dictionary that given two key-values:
            data_path: the evaluated dir
            log: the log of this dir
    '''
    model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)

    log_list = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  Now Evaluation >>>>")
        print(">>>>  OUTPUT: {}".format(ouput_root))
        print(">>>>  GT: {}".format(gt_root))
        _, log = calc_image_LPIPS(ouput_root, gt_root, model=model, use_gpu=use_gpu, spatial=spatial)
        log_list.append({
            'data_path': ouput_root,
            'log': log
        })

    print("--------------------------------------------------------------------------------------")
    for i, log in enumerate(log_list):
        print("## The {}-th:".format(i))
        print(">> ", log['data_path'])
        print(">> ", log['log'])

    return log_list

import os
import cv2
import numpy as np
from base.os_base import handle_dir, get_fname_ext, listdir


def batch_cv2_resize_images(ori_root, dest_root, scale=1.0, method='bicubic', filename_template="{}.png"):
    '''
    function:
        resizing images in batches
    params:
        ori_root: string, the dir of images that need to be processed
        dest_root: string, the dir to save processed images
        scale: float, the resize scale
        method: string, the interpolation method,
            optional: 'nearest', 'bilinear', 'bicubic'
            default: 'bicubic'
        filename_template: string, the filename template for saving images
    '''
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        raise Exception('Unknown method!')

    handle_dir(dest_root)
    scale = float(scale)
    images_fname = sorted(listdir(ori_root))
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf)).astype('float32')
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)
        cv2.imwrite(os.path.join(dest_root, filename_template.format(get_fname_ext(imf)[0])), img)
        print("Image", imf, "resize done !")


def margin_patch(patch, margin_width=5):
    '''margin patch by red cycle'''
    red = np.zeros_like(patch)
    red[:, :, 1] = 255
    mask = np.zeros_like(patch)
    mask[:margin_width, :, :] = mask[-margin_width:, :, :] = mask[:, :margin_width, :] = mask[:, -margin_width:, :] = 1
    res = red * mask + patch * (1 - mask)
    return res


def circle_zoom_img(img, pos_1, pos_2, scale=2., hr_pos='right_down'):
    '''circle and zoom a patch in image'''
    patch_lr = img[pos_1[0]:pos_1[1], pos_2[0]:pos_2[1], :]
    patch_hr = cv2.resize(patch_lr, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    patch_lr = margin_patch(patch_lr)
    patch_hr = margin_patch(patch_hr)
    img[pos_1[0]:pos_1[1], pos_2[0]:pos_2[1], :] = patch_lr
    if hr_pos == 'right_down':
        img[-patch_hr.shape[0]:, -patch_hr.shape[1]:, :] = patch_hr
    elif hr_pos == 'left_down':
        img[-patch_hr.shape[0]:, :patch_hr.shape[1], :] = patch_hr
    return img


def circle_img(img, pos_1, pos_2, margin_width=2):
    '''circle a patch in image'''
    patch_lr = img[pos_1[0]:pos_1[1], pos_2[0]:pos_2[1], :]
    patch_lr = margin_patch(patch_lr, margin_width=margin_width)
    img[pos_1[0]:pos_1[1], pos_2[0]:pos_2[1], :] = patch_lr
    return img

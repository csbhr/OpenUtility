'''
require:
    filenames:
        filenames should not contrain symbol "-"
    crop flags:
        the crop flag "x-x-x-x" is at the end of filename when cropping
        so, the crop flag should be at the end of filename when combining

'''

import cv2
import os
import numpy as np
from base.image_base import evaluate_smooth
from base.os_base import handle_dir, listdir, glob_match


def crop_img_with_padding(img, min_size=(100, 100), padding=10):
    h, w, c = img.shape
    n_x, n_y = h // min_size[0], w // min_size[1]

    croped_imgs = {}
    for i in range(n_x):
        for j in range(n_y):
            xl, xr = i * min_size[0] - padding, (i + 1) * min_size[0] + padding
            yl, yr = j * min_size[0] - padding, (j + 1) * min_size[0] + padding
            if i == 0:
                xl = 0
            if i == n_x - 1:
                xr = h
            if j == 0:
                yl = 0
            if j == n_y - 1:
                yr = w
            croped_imgs['{}-{}-{}-{}'.format(n_x, n_y, i, j)] = img[xl:xr, yl:yr, :]

    return croped_imgs


def combine_img(croped_imgs, padding=10):
    keys = list(croped_imgs.keys())
    n_x, n_y = int(keys[0].split('-')[0]), int(keys[0].split('-')[1])
    img_blocks = [["" for _ in range(n_y)] for _ in range(n_x)]
    for k in keys:
        i, j = int(k.split('-')[2]), int(k.split('-')[3])
        xl, xr, yl, yr = padding, -padding, padding, -padding
        if i == 0:
            xl = 0
        if i == n_x - 1:
            xr = 9999
        if j == 0:
            yl = 0
        if j == n_y - 1:
            yr = 9999
        img_blocks[i][j] = croped_imgs[k][xl:xr, yl:yr, :]

    line_imgs = []
    for i in range(n_x):
        img_line = img_blocks[i][0]
        for j in range(n_y - 1):
            img_line = np.concatenate((img_line, img_blocks[i][j + 1]), axis=1)
        line_imgs.append(img_line)
    combined_img = line_imgs[0]
    for i in range(n_x - 1):
        combined_img = np.concatenate((combined_img, line_imgs[i + 1]), axis=0)
    return combined_img


def traverse_crop_img(img, dsize=(100, 100), interval=10):
    h, w, c = img.shape

    croped_imgs = []
    for i in range(9999):
        isbreak_x = False
        ix = i * interval
        xl, xr = ix, ix + dsize[0]
        if xr > h:
            xl, xr = h - dsize[0], h
            isbreak_x = True
        for j in range(9999):
            isbreak_y = False
            iy = j * interval
            yl, yr = iy, iy + dsize[1]
            if yr > w:
                yl, yr = w - dsize[1], w
                isbreak_y = True
            croped = img[xl:xr, yl:yr, :]
            croped_imgs.append(croped)
            if isbreak_y:
                break
        if isbreak_x:
            break

    return croped_imgs


def batch_crop_img_with_padding(ori_root, dest_root, min_size=(100, 100), padding=10):
    '''
    function:
        cropping image to many patches with padding
        it can be used for inferring large image
    params:
        ori_root: the dir of images that need to be processed
        dest_root: the dir to save processed images
        min_size: a tuple (h, w) the min size of crop, the border patch will be larger
        padding: the padding size of each patch
    notice:
        filenames should not contain the character "-"
        the crop flag "x-x-x-x" will be at the end of filename when cropping
    '''
    handle_dir(dest_root)
    images_fname = sorted(listdir(ori_root))
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf))
        img_cropped = crop_img_with_padding(img, min_size=min_size, padding=padding)
        for k in img_cropped.keys():
            cv2.imwrite(os.path.join(dest_root, "{}_{}.png".format(os.path.basename(imf).split('.')[0], k)),
                        img_cropped[k])
        print(imf, "crop done !")


def batch_combine_img(ori_root, dest_root, padding=10):
    '''
    function:
        combining many patches to image
        it can be used to combine patches to image, when you finish inferring large image with cropped patches
    params:
        ori_root: the dir of images that need to be processed
        dest_root: the dir to save processed images
        padding: the padding size of each patch
    notice:
        filenames should not contain the character "-" except for the crop flag
        the crop flag "x-x-x-x" should be at the end of filename when combining
    '''
    handle_dir(dest_root)
    images_fname = [fn[:-(len(fn.split('_')[-1]) + 1)] for fn in listdir(ori_root)]
    images_fname = list(set(images_fname))
    for imf in images_fname:
        croped_imgs_path = sorted(glob_match(os.path.join(ori_root, "{}*".format(imf))))
        croped_imgs = {}
        for cip in croped_imgs_path:
            img = cv2.imread(cip)
            k = cip.split('.')[0].split('_')[-1]
            croped_imgs[k] = img
        img_combined = combine_img(croped_imgs, padding=padding)
        cv2.imwrite(os.path.join(dest_root, "{}.png".format(imf)), img_combined)
        print("{}.png".format(imf), "combine done !")


def batch_traverse_crop_img(ori_root, dest_root, dsize=(100, 100), interval=10):
    '''
    function:
        traversing crop image to many patches with same interval
    params:
        ori_root: the dir of images that need to be processed
        dest_root: the dir to save processed images
        dsize: a tuple (h, w) the size of crop, the border patch will be overlapped for satisfing the dsize
        interval: the interval when traversing
    '''
    handle_dir(dest_root)
    images_fname = sorted(listdir(ori_root))
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf))
        img_cropped = traverse_crop_img(img, dsize=dsize, interval=interval)
        for i, cim in enumerate(img_cropped):
            cv2.imwrite(os.path.join(dest_root, "{}_{}.png".format(os.path.basename(imf).split('.')[0], i)), cim)
        print(imf, "crop done !")


def batch_select_valid_patch(ori_root, dest_root, thre=7):
    '''
    function:
        selecting valid patch that are not too smooth
    params:
        ori_root: the dir of patches that need to be selected
        dest_root: the dir to save selected patch
        thre: the threshold value of smooth
    '''
    handle_dir(dest_root)
    images_fname = sorted(listdir(ori_root))
    total_num = len(images_fname)
    valid_num = 0
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf))
        smooth = evaluate_smooth(img)
        if smooth > thre:
            cv2.imwrite(os.path.join(dest_root, imf), img)
            valid_num += 1
        else:
            print(imf, "too smooth, smooth={}".format(smooth))
    print("Total {} patches, valid {}, remove {}".format(total_num, valid_num, total_num - valid_num))

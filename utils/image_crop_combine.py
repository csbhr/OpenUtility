'''
require:
    filenames:
        filenames should not contrain symbol "-"
    crop flags:
        the crop flag "x-x-x-x" is at the end of filename when cropping
        so, the crop flag should be at the end of filename when comblning

'''

import cv2
import os
import glob
import numpy as np


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


def crop_img(img, min_size=(100, 100), padding=10):
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


def batch_crop_img(ori_root, dest_root, min_size=(100, 100), padding=10):
    handle_dir(dest_root)
    images_fname = sorted(os.listdir(ori_root))
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf))
        img_cropped = crop_img(img, min_size=min_size, padding=padding)
        for k in img_cropped.keys():
            cv2.imwrite(os.path.join(dest_root, "{}_{}.png".format(os.path.basename(imf).split('.')[0], k)),
                        img_cropped[k])
        print(imf, "crop done !")


def batch_combine_img(ori_root, dest_root, padding=10):
    handle_dir(dest_root)
    images_fname = [fn[:-(len(fn.split('_')[-1]) + 1)] for fn in os.listdir(ori_root)]
    images_fname = list(set(images_fname))
    for imf in images_fname:
        croped_imgs_path = sorted(glob.glob(os.path.join(ori_root, "{}*".format(imf))))
        croped_imgs = {}
        for cip in croped_imgs_path:
            img = cv2.imread(cip)
            k = cip.split('.')[0].split('_')[-1]
            croped_imgs[k] = img
        img_combined = combine_img(croped_imgs, padding=padding)
        cv2.imwrite(os.path.join(dest_root, "{}.png".format(imf)), img_combined)
        print("{}.png".format(imf), "combine done !")

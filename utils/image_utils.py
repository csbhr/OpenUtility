import os
import glob
import cv2


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


def resize_images(ori_root, save_root, scale):
    handle_dir(save_root)
    imgs_path = sorted(glob.glob(os.path.join(ori_root, "*")))
    for ip in imgs_path:
        img_name = os.path.basename(ip)
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        bicubic_img = cv2.resize(img, (w * scale, h * scale), cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_root, img_name), bicubic_img)
        print('resize img:', ip)

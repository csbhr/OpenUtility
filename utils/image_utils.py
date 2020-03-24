import os
import cv2
from utils.base_utils import handle_dir, get_fname_ext


def batch_resize_images(ori_root, dest_root, scale=1.0, postfix=True):
    '''
    function:
        resizing images in batches
    params:
        ori_root: the dir of images that need to be processed
        dest_root: the dir to save processed images
        scale: float, the resize scale
        postfix: if True: add scale as postfix for filename after resizing
    '''
    handle_dir(dest_root)
    scale = float(scale)
    if postfix:
        pf = "{}d{}".format(str(scale).split(".")[0], str(scale).split(".")[1])
        dfile_template = "{}_"+pf+".png"
    else:
        dfile_template = "{}.png"
    images_fname = sorted(os.listdir(ori_root))
    for imf in images_fname:
        img = cv2.imread(os.path.join(ori_root, imf))
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dest_root, dfile_template.format(get_fname_ext(imf)[0])), img)
        print("image", imf, "resize done !")

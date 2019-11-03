from PIL import Image
import os
import numpy
from scipy import misc
import glob


def resize_image(ori_root, save_root, scale):
    videos_path = sorted(glob.glob(os.path.join(ori_root, "*")))

    for vp in videos_path:
        video_name = os.path.basename(vp)
        imgs_path = sorted(glob.glob(os.path.join(vp, "*")))

        for ip in imgs_path:
            print(ip)

            img_name = os.path.basename(ip)
            img = numpy.array(Image.open(ip))
            bicubic_img = misc.imresize(img, scale, interp='bicubic')

            savepath = os.path.join(save_root, video_name)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            misc.imsave(os.path.join(savepath, img_name), bicubic_img)
'''
require:
    filenames:
        filenames should not contrain symbol "-"
    crop flags:
        the crop flag "x-x-x-x" is at the end of filename when cropping
        so, the crop flag should be at the end of filename when comblning

'''

from utils.image_crop_combine import *

# demo crop images
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_crop_img(ori_root, dest_root, min_size=(800, 800), padding=100)

# demo combine images
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_combine_img(ori_root, dest_root, padding=100)
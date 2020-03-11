from utils.calc_image_psnr_ssim import *

'''

Demo to calculate images' PSNR/SSIM

root_list: Dictionary
    each item should given two key-values:
        output: the dir of output images
        gt: the dir of gt images
        
batch_calc_image_PSNR_SSIM(): Function
    required params:
        root_list
    optional params:
        crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
        test_ycbcr: default=False, if True, applying Ycbcr color space

'''

root_list = [
    {
        'output': '/path/to/output images',
        'gt': '/path/to/gt images'
    },
    {
        'output': '/path/to/output images',
        'gt': '/path/to/gt images'
    },
]

batch_calc_image_PSNR_SSIM(root_list)

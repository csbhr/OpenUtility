from utils.calc_video_psnr_ssim import *

'''

Demo to calculate videos' PSNR/SSIM

root_list: Dictionary
    each item should given two key-values:
        output: the dir of output videos
        gt: the dir of gt videos
        
save_csv_log_root: String
    thr dir of output log

batch_calc_video_PSNR_SSIM_toCSV(): Function
    required params:
        root_list
        save_csv_log_root
    optional params:
        crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
        test_ycbcr: default=False, if True, applying Ycbcr color space
        combine_save: default=False, if True, combining all output log to one csv file
        match_byname: default=False, if True, matching output video and gt video by filename

'''

root_list = [
    {
        'output': '/path/to/output videos',
        'gt': '/path/to/gt videos'
    },
    {
        'output': '/path/to/output videos',
        'gt': '/path/to/gt videos'
    },
]
save_csv_log_root = '/path/to/log'

batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_log_root)

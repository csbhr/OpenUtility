from utils.calc_video_psnr_ssim import *

root_list = [
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/DUF',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/TOFlow',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/RBPN',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/EDVR',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/ours',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/ours_deconv_finetune',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/Vid4/ours_wo_deconv',
        'gt': '/home/csbhr/workspace/sr_result/Vid4/GT'
    }
]
save_csv_log_root = '/home/csbhr/workspace/sr_result/Vid4'

batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_log_root, crop_border=7, test_ycbcr=True, combine_save=True)

################################################################################################################
root_list = [
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/DUF',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/TOFlow',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/RBPN',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/EDVR',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/ours',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/ours_deconv_finetune',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/REDS4/ours_wo_deconv',
        'gt': '/home/csbhr/workspace/sr_result/REDS4/GT'
    }
]
save_csv_log_root = '/home/csbhr/workspace/sr_result/REDS4'

batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_log_root, crop_border=7, test_ycbcr=True, combine_save=True)

################################################################################################################
root_list = [
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/DUF',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/TOFlow',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/RBPN',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/EDVR',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/ours',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/ours_deconv_finetune',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    },
    {
        'output': '/home/csbhr/workspace/sr_result/SPMC/ours_wo_deconv',
        'gt': '/home/csbhr/workspace/sr_result/SPMC/GT'
    }
]
save_csv_log_root = '/home/csbhr/workspace/sr_result/SPMC'

batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_log_root, crop_border=7, test_ycbcr=True, combine_save=True)

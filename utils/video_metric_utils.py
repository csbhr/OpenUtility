import os
import numpy as np
from base.file_io_base import write_csv
from base.os_base import listdir
import utils.PerceptualSimilarity.models as lpips_models
from utils.image_metric_utils import calc_image_PSNR_SSIM, calc_image_LPIPS


def calc_video_PSNR_SSIM(output_root, gt_root, crop_border=4, shift_window_size=0, test_ycbcr=False, crop_GT=False):
    '''
    计算视频的 PSNR、SSIM，使用 EDVR 的计算方式
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    PSNR_sum = 0.
    SSIM_sum = 0.
    img_num = 0

    video_PSNR = []
    video_SSIM = []

    video_list = sorted(listdir(output_root))
    for v in video_list:
        v_PSNR_list, v_SSIM_list, _ = calc_image_PSNR_SSIM(
            output_root=os.path.join(output_root, v),
            gt_root=os.path.join(gt_root, v),
            crop_border=crop_border, shift_window_size=shift_window_size,
            test_ycbcr=test_ycbcr, crop_GT=crop_GT
        )
        PSNR_sum += sum(v_PSNR_list)
        SSIM_sum += sum(v_SSIM_list)
        img_num += len(v_PSNR_list)

        video_PSNR.append({
            'video_name': v,
            'psnr': v_PSNR_list
        })
        video_SSIM.append({
            'video_name': v,
            'ssim': v_SSIM_list
        })

    logs = []
    PSNR_SSIM_csv_log = {
        'col_names': [],
        'row_names': [output_root],
        'psnr_ssim': [[]]
    }
    for v_psnr, v_ssim in zip(video_PSNR, video_SSIM):
        PSNR_SSIM_csv_log['col_names'].append('#{}'.format(v_psnr['video_name']))
        PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                                      sum(v_ssim['ssim']) / len(v_ssim['ssim'])))
        log = 'Video: {} PSNR={:.5}, SSIM={:.4}'.format(v_psnr['video_name'],
                                                        sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                        sum(v_ssim['ssim']) / len(v_ssim['ssim']))
        print(log)
        logs.append(log)
    PSNR_SSIM_csv_log['col_names'].append('AVG')
    PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num))
    log = 'Average PSNR={:.5}, SSIM={:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num)
    print(log)
    logs.append(log)

    return PSNR_SSIM_csv_log, logs


def batch_calc_video_PSNR_SSIM(root_list, crop_border=4, shift_window_size=0, test_ycbcr=False, crop_GT=False,
                               save_log=False, save_log_root=None, combine_save=False):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output videos
            gt: the dir of gt videos
    optional params:
        crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
        shift_window_size: defalut=0, if >0, shifting image within a window for best metric
        test_ycbcr: default=False, if True, applying Ycbcr color space
        crop_GT: default=False, if True, cropping GT to output size
        save_log: default=False, if True, saving csv log
        save_log_root: thr dir of output log
        combine_save: default=False, if True, combining all output log to one csv file
    return:
        log_list: a list, each item is a dictionary that given two key-values:
            data_path: the evaluated dir
            log: the log of this dir
    '''
    if save_log:
        assert save_log_root is not None, "Unknown save_log_root!"

    total_csv_log = []
    log_list = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  Now Evaluation >>>>")
        print(">>>>  OUTPUT: {}".format(ouput_root))
        print(">>>>  GT: {}".format(gt_root))
        csv_log, logs = calc_video_PSNR_SSIM(
            ouput_root, gt_root, crop_border=crop_border, shift_window_size=shift_window_size,
            test_ycbcr=test_ycbcr, crop_GT=crop_GT
        )
        log_list.append({
            'data_path': ouput_root,
            'log': logs
        })

        # output the PSNR/SSIM log of each evaluated dir to a single csv file
        if save_log:
            csv_log['row_names'] = [os.path.basename(p) for p in csv_log['row_names']]
            write_csv(file_path=os.path.join(save_log_root, "{}_{}.csv".format(i, csv_log['row_names'][0])),
                      data=np.array(csv_log['psnr_ssim']),
                      row_names=csv_log['row_names'],
                      col_names=csv_log['col_names'])
            total_csv_log.append(csv_log)

    # output all PSNR/SSIM log to a csv file
    if save_log and combine_save and len(total_csv_log) > 0:
        com_csv_log = {
            'col_names': total_csv_log[0]['col_names'],
            'row_names': [],
            'psnr_ssim': []
        }
        for csv_log in total_csv_log:
            com_csv_log['row_names'].append(csv_log['row_names'][0])
            com_csv_log['psnr_ssim'].append(csv_log['psnr_ssim'][0])
        write_csv(file_path=os.path.join(save_log_root, "psnr_ssim.csv"),
                  data=np.array(com_csv_log['psnr_ssim']),
                  row_names=com_csv_log['row_names'],
                  col_names=com_csv_log['col_names'])

    print("--------------------------------------------------------------------------------------")
    for i, logs in enumerate(log_list):
        print("## The {}-th:".format(i))
        print(">> ", logs['data_path'])
        for log in logs['log']:
            print(">> ", log)

    return log_list


def calc_video_LPIPS(output_root, gt_root, model=None, use_gpu=False, spatial=True):
    '''
    计算视频的 LPIPS
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    if model is None:
        model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)

    LPIPS_sum = 0.
    img_num = 0

    video_LPIPS = []

    video_list = sorted(listdir(output_root))
    for v in video_list:
        v_LPIPS_list, _ = calc_image_LPIPS(
            output_root=os.path.join(output_root, v),
            gt_root=os.path.join(gt_root, v),
            model=model, use_gpu=use_gpu, spatial=spatial
        )
        LPIPS_sum += sum(v_LPIPS_list)
        img_num += len(v_LPIPS_list)

        video_LPIPS.append({
            'video_name': v,
            'lpips': v_LPIPS_list
        })

    logs = []
    LPIPS_csv_log = {
        'col_names': [],
        'row_names': [output_root],
        'lpips': [[]]
    }
    for v_lpips in video_LPIPS:
        LPIPS_csv_log['col_names'].append('#{}'.format(v_lpips['video_name']))
        LPIPS_csv_log['lpips'][0].append('{:.4}'.format(sum(v_lpips['lpips']) / len(v_lpips['lpips'])))
        log = 'Video: {} LPIPS={:.4}'.format(v_lpips['video_name'], sum(v_lpips['lpips']) / len(v_lpips['lpips']))
        print(log)
        logs.append(log)
    LPIPS_csv_log['col_names'].append('AVG')
    LPIPS_csv_log['lpips'][0].append('{:.4}'.format(LPIPS_sum / img_num))
    log = 'Average LPIPS={:.4}'.format(LPIPS_sum / img_num)
    print(log)
    logs.append(log)

    return LPIPS_csv_log, logs


def batch_calc_video_LPIPS(root_list, use_gpu=False, spatial=True,
                           save_log=False, save_log_root=None, combine_save=False):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output videos
            gt: the dir of gt videos
    optional params:
        use_gpu: defalut=False, if True, using gpu
        spatial: default=True, if True, return spatial map
        save_log: default=False, if True, saving csv log
        save_log_root: thr dir of output log
        combine_save: default=False, if True, combining all output log to one csv file
    return:
        log_list: a list, each item is a dictionary that given two key-values:
            data_path: the evaluated dir
            log: the log of this dir
    '''
    if save_log:
        assert save_log_root is not None, "Unknown save_log_root!"

    model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)

    total_csv_log = []
    log_list = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  Now Evaluation >>>>")
        print(">>>>  OUTPUT: {}".format(ouput_root))
        print(">>>>  GT: {}".format(gt_root))
        csv_log, logs = calc_video_LPIPS(ouput_root, gt_root, model=model, use_gpu=use_gpu, spatial=spatial)
        log_list.append({
            'data_path': ouput_root,
            'log': logs
        })

        # output the LPIPS log of each evaluated dir to a single csv file
        if save_log:
            csv_log['row_names'] = [os.path.basename(p) for p in csv_log['row_names']]
            write_csv(file_path=os.path.join(save_log_root, "{}_{}.csv".format(i, csv_log['row_names'][0])),
                      data=np.array(csv_log['lpips']),
                      row_names=csv_log['row_names'],
                      col_names=csv_log['col_names'])
            total_csv_log.append(csv_log)

    # output all LPIPS log to a csv file
    if save_log and combine_save and len(total_csv_log) > 0:
        com_csv_log = {
            'col_names': total_csv_log[0]['col_names'],
            'row_names': [],
            'lpips': []
        }
        for csv_log in total_csv_log:
            com_csv_log['row_names'].append(csv_log['row_names'][0])
            com_csv_log['lpips'].append(csv_log['lpips'][0])
        write_csv(file_path=os.path.join(save_log_root, "lpips.csv"),
                  data=np.array(com_csv_log['lpips']),
                  row_names=com_csv_log['row_names'],
                  col_names=com_csv_log['col_names'])

    print("--------------------------------------------------------------------------------------")
    for i, logs in enumerate(log_list):
        print("## The {}-th:".format(i))
        print(">> ", logs['data_path'])
        for log in logs['log']:
            print(">> ", log)

    return log_list

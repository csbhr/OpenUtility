import os
import cv2
import glob
from utils import base_utils
from utils.file_io_utils import write_csv


def calc_video_PSNR_SSIM(ouput_root, gt_root, crop_border=4, test_ycbcr=False):
    '''
    计算视频的 PSNR、SSIM，使用 EDVR 的计算方式
    要求 ouput_root, gt_root 中的文件按顺序一一对应
    '''

    if test_ycbcr:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    PSNR_sum = 0.
    SSIM_sum = 0.
    img_num = 0

    video_PSNR = []
    video_SSIM = []

    video_list = sorted(os.listdir(ouput_root))
    for v in video_list:

        video_PSNR.append({
            'video_name': v,
            'psnr': []
        })
        video_SSIM.append({
            'video_name': v,
            'ssim': []
        })

        output_img_list = sorted(os.listdir(os.path.join(ouput_root, v)))
        gt_img_list = sorted(os.listdir(os.path.join(gt_root, v)))
        for o_im, g_im in zip(output_img_list, gt_img_list):
            o_im_path = os.path.join(ouput_root, v, o_im)
            g_im_path = os.path.join(gt_root, v, g_im)
            im_GT = cv2.imread(g_im_path) / 255.
            im_Gen = cv2.imread(o_im_path) / 255.

            if test_ycbcr and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT = base_utils.bgr2ycbcr(im_GT)
                im_Gen = base_utils.bgr2ycbcr(im_Gen)

            # crop borders
            if im_GT.ndim == 3:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT.ndim == 2:
                cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT.ndim))

            psnr = base_utils.PSNR_EDVR(cropped_GT * 255, cropped_Gen * 255)
            ssim = base_utils.SSIM_EDVR(cropped_GT * 255, cropped_Gen * 255)
            PSNR_sum += psnr
            SSIM_sum += ssim
            img_num += 1
            video_PSNR[-1]['psnr'].append(psnr)
            video_SSIM[-1]['ssim'].append(ssim)

            print("{}-{} PSNR={:.5}, SSIM={:.4}".format(v, o_im, psnr, ssim))

    PSNR_SSIM_csv_log = {
        'col_names': [],
        'row_names': [ouput_root],
        'psnr_ssim': [[]]
    }
    for v_psnr, v_ssim in zip(video_PSNR, video_SSIM):
        PSNR_SSIM_csv_log['col_names'].append('#{}'.format(v_psnr['video_name']))
        PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                                      sum(v_ssim['ssim']) / len(v_ssim['ssim'])))
        print('Video: {} PSNR={:.5}, SSIM={:.4}'.format(v_psnr['video_name'],
                                                        sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                        sum(v_ssim['ssim']) / len(v_ssim['ssim'])))
    PSNR_SSIM_csv_log['col_names'].append('AVG')
    PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num))
    print('Average PSNR={:.5}, SSIM={:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num))

    return PSNR_SSIM_csv_log


def calc_video_PSNR_SSIM_byName(ouput_root, gt_root, crop_border=4, test_ycbcr=False):
    '''
    计算视频的 PSNR、SSIM，使用 EDVR 的计算方式
    要求 ouput_root, gt_root 中的成对的文件名要一致
    '''

    if test_ycbcr:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    PSNR_sum = 0.
    SSIM_sum = 0.
    img_num = 0

    video_PSNR = []
    video_SSIM = []

    video_list = sorted(os.listdir(ouput_root))
    for v in video_list:

        video_PSNR.append({
            'video_name': v,
            'psnr': []
        })
        video_SSIM.append({
            'video_name': v,
            'ssim': []
        })

        output_img_list = sorted(os.listdir(os.path.join(ouput_root, v)))
        for o_im in output_img_list:
            o_im_path = os.path.join(ouput_root, v, o_im)
            o_im_fname = o_im[:-(len(o_im.split('.')[-1]) + 1)]
            g_im_path = glob.glob(os.path.join(gt_root, v, '{}.*'.format(o_im_fname)))[0]
            im_GT = cv2.imread(g_im_path) / 255.
            im_Gen = cv2.imread(o_im_path) / 255.

            if test_ycbcr and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT = base_utils.bgr2ycbcr(im_GT)
                im_Gen = base_utils.bgr2ycbcr(im_Gen)

            # crop borders
            if crop_border != 0:
                if im_GT.ndim == 3:
                    cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
                    cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
                elif im_GT.ndim == 2:
                    cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]
                else:
                    raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT.ndim))
            else:
                cropped_GT = im_GT
                cropped_Gen = im_Gen

            psnr = base_utils.PSNR_EDVR(cropped_GT * 255, cropped_Gen * 255)
            ssim = base_utils.SSIM_EDVR(cropped_GT * 255, cropped_Gen * 255)
            PSNR_sum += psnr
            SSIM_sum += ssim
            img_num += 1
            video_PSNR[-1]['psnr'].append(psnr)
            video_SSIM[-1]['ssim'].append(ssim)

            print("{}-{} PSNR={:.5}, SSIM={:.4}".format(v, o_im, psnr, ssim))

    PSNR_SSIM_csv_log = {
        'col_names': [],
        'row_names': [ouput_root],
        'psnr_ssim': [[]]
    }
    for v_psnr, v_ssim in zip(video_PSNR, video_SSIM):
        PSNR_SSIM_csv_log['col_names'].append(v_psnr['video_name'])
        PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                                      sum(v_ssim['ssim']) / len(v_ssim['ssim'])))
        print('Video: {} PSNR={:.5}, SSIM={:.4}'.format(v_psnr['video_name'],
                                                        sum(v_psnr['psnr']) / len(v_psnr['psnr']),
                                                        sum(v_ssim['ssim']) / len(v_ssim['ssim'])))
    PSNR_SSIM_csv_log['col_names'].append('AVG')
    PSNR_SSIM_csv_log['psnr_ssim'][0].append('{:.5}/{:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num))
    print('Average PSNR={:.5}, SSIM={:.4}'.format(PSNR_sum / img_num, SSIM_sum / img_num))

    return PSNR_SSIM_csv_log


def batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_root, crop_border=4, test_ycbcr=False,
                                     combine_save=False, match_byname=False):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output videos
            gt: the dir of gt videos
        save_csv_log_root: thr dir of output log
    optional params:
        crop_border: defalut=4, crop pixels when calculating PSNR/SSIM
        test_ycbcr: default=False, if True, applying Ycbcr color space
        combine_save: default=False, if True, combining all output log to one csv file
        match_byname: default=False, if True, matching output video and gt video by filename
    '''
    total_csv_log = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  now test >>>>")
        print(">>>>  output: {}".format(ouput_root))
        print(">>>>  gt: {}".format(gt_root))
        if match_byname:
            csv_log = calc_video_PSNR_SSIM_byName(ouput_root, gt_root, crop_border=crop_border, test_ycbcr=test_ycbcr)
        else:
            csv_log = calc_video_PSNR_SSIM(ouput_root, gt_root, crop_border=crop_border, test_ycbcr=test_ycbcr)

        # output the PSNR/SSIM log of each evaluated dir to a single csv file
        csv_log['row_names'] = [os.path.basename(p) for p in csv_log['row_names']]
        write_csv(file_path=os.path.join(save_csv_root, "{}_{}.csv".format(i, csv_log['row_names'][0])),
                  data=csv_log['psnr_ssim'],
                  row_names=csv_log['row_names'],
                  col_names=csv_log['col_names'])
        total_csv_log.append(csv_log)

    # output all PSNR/SSIM log to a csv file
    if combine_save and len(total_csv_log) > 0:
        com_csv_log = {
            'col_names': total_csv_log[0]['col_names'],
            'row_names': [],
            'psnr_ssim': []
        }
        for csv_log in total_csv_log:
            com_csv_log['row_names'].append(csv_log['row_names'][0])
            com_csv_log['psnr_ssim'].append(csv_log['psnr_ssim'][0])
        write_csv(file_path=os.path.join(save_csv_root, "psnr_ssim.csv"),
                  data=com_csv_log['psnr_ssim'],
                  row_names=com_csv_log['row_names'],
                  col_names=com_csv_log['col_names'])

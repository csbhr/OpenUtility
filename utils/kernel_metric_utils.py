import os
from base import kernel_base
from base.kernel_base import load_mat_kernel
from base.os_base import listdir


def calc_kernel_metric(output_root, gt_root):
    '''
    计算 kernel 的 metric: gradient similarity, rmse, psnr, ssim
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    metric_dict = {
        'gradient_similarity': [],
        'rmse': [],
        'psnr': [],
        'ssim': [],
    }

    output_kernel_list = sorted(listdir(output_root))
    gt_kernel_list = sorted(listdir(gt_root))
    for o_k, g_k in zip(output_kernel_list, gt_kernel_list):
        o_k_path = os.path.join(output_root, o_k)
        g_k_path = os.path.join(gt_root, g_k)
        kernel_GT = load_mat_kernel(g_k_path)
        kernel_Gen = load_mat_kernel(o_k_path)

        gradient_similarity = kernel_base.Gradient_Similarity(kernel_Gen, kernel_GT)
        rmse, psnr, ssim = kernel_base.Kernel_RMSE_PSNR_SSIM(kernel_Gen, kernel_GT)
        metric_dict['gradient_similarity'].append(gradient_similarity)
        metric_dict['rmse'].append(rmse)
        metric_dict['psnr'].append(psnr)
        metric_dict['ssim'].append(ssim)

        print("{} Gradient-Similarity={:.4}, RMSE={:.4}, PSNR={:.4}, SSIM={:.4}".format(
            o_k, gradient_similarity, rmse, psnr, ssim
        ))

    log = 'Average Gradient-Similarity={:.4}, RMSE={:.4}, PSNR={:.4}, SSIM={:.4}'.format(
        sum(metric_dict['gradient_similarity']) / len(metric_dict['gradient_similarity']),
        sum(metric_dict['rmse']) / len(metric_dict['rmse']),
        sum(metric_dict['psnr']) / len(metric_dict['psnr']),
        sum(metric_dict['ssim']) / len(metric_dict['ssim'])
    )
    print(log)

    return metric_dict, log


def calc_kernel_metric_video(output_root, gt_root):
    '''
    计算视频的 kernel 的 metric: gradient similarity, rmse
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''
    gradient_similarity_sum = 0.
    rmse_sum = 0.
    psnr_sum = 0.
    ssim_sum = 0.
    kernel_num = 0

    video_metric = []

    video_list = sorted(listdir(output_root))
    for v in video_list:
        v_metric_list, _ = calc_kernel_metric(
            output_root=os.path.join(output_root, v),
            gt_root=os.path.join(gt_root, v)
        )
        gradient_similarity_sum += sum(v_metric_list['gradient_similarity'])
        rmse_sum += sum(v_metric_list['rmse'])
        psnr_sum += sum(v_metric_list['psnr'])
        ssim_sum += sum(v_metric_list['ssim'])
        kernel_num += len(v_metric_list['gradient_similarity'])

        video_metric.append({
            'video_name': v,
            'gradient_similarity': v_metric_list['gradient_similarity'],
            'rmse': v_metric_list['rmse'],
            'psnr': v_metric_list['psnr'],
            'ssim': v_metric_list['ssim']
        })

    logs = []
    for v_m in video_metric:
        log = 'Video: {} Gradient-Similarity={:.4}, RMSE={:.4}, PSNR={:.4}, SSIM={:.4}'.format(
            v_m['video_name'],
            sum(v_m['gradient_similarity']) / len(v_m['gradient_similarity']),
            sum(v_m['rmse']) / len(v_m['rmse']),
            sum(v_m['psnr']) / len(v_m['psnr']),
            sum(v_m['ssim']) / len(v_m['ssim'])
        )
        print(log)
        logs.append(log)
    log = 'Average Gradient-Similarity={:.4}, RMSE={:.4}, PSNR={:.4}, SSIM={:.4}'.format(
        gradient_similarity_sum / kernel_num,
        rmse_sum / kernel_num,
        psnr_sum / kernel_num,
        ssim_sum / kernel_num
    )
    print(log)
    logs.append(log)

    return video_metric, logs


def batch_calc_kernel_metric(root_list, video_type=False):
    '''
    required params:
        root_list: a list, each item should be a dictionary that given two key-values:
            output: the dir of output images
            gt: the dir of gt images
    return:
        log_list: a list, each item is a dictionary that given two key-values:
            data_path: the evaluated dir
            log: the log of this dir
    '''
    log_list = []
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  Now Evaluation >>>>")
        print(">>>>  OUTPUT: {}".format(ouput_root))
        print(">>>>  GT: {}".format(gt_root))
        if video_type:
            _, log = calc_kernel_metric_video(ouput_root, gt_root)
        else:
            _, log = calc_kernel_metric(ouput_root, gt_root)
        log_list.append({
            'data_path': ouput_root,
            'log': log
        })

    print("--------------------------------------------------------------------------------------")
    for i, log in enumerate(log_list):
        print("## The {}-th:".format(i))
        print(">> ", log['data_path'])
        if video_type:
            for lo in log['log']:
                print(">> ", lo)
        else:
            print(">> ", log['log'])

    return log_list

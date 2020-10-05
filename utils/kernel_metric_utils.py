import os
from base import kernel_base
from base.os_base import listdir
import scipy.io as scio


def load_mat_kernel(mat_path):
    data_dict = scio.loadmat(mat_path)
    key = [v for v in data_dict.keys() if v not in ['__header__', '__version__', '__globals__']][0]
    return data_dict[key]


def calc_kernel_gradient_similarity(output_root, gt_root):
    '''
    计算 kernel 的 gradient similarity
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''

    GS_list = []
    output_kernel_list = sorted(listdir(output_root))
    gt_kernel_list = sorted(listdir(gt_root))
    for o_k, g_k in zip(output_kernel_list, gt_kernel_list):
        o_k_path = os.path.join(output_root, o_k)
        g_k_path = os.path.join(gt_root, g_k)
        kernel_GT = load_mat_kernel(g_k_path)
        kernel_Gen = load_mat_kernel(o_k_path)

        gs = kernel_base.Gradient_Similarity(kernel_Gen, kernel_GT)
        GS_list.append(gs)

        print("{} Gradient-Similarity={:.4}".format(o_k, gs))

    log = 'Average Gradient-Similarity={:.4}'.format(sum(GS_list) / len(GS_list))
    print(log)

    return GS_list, log


def calc_kernel_gradient_similarity_video(output_root, gt_root):
    '''
    计算视频的 kernel 的 gradient similarity
    要求 output_root, gt_root 中的文件按顺序一一对应
    '''
    GS_sum = 0.
    kernel_num = 0

    video_GS = []

    video_list = sorted(listdir(output_root))
    for v in video_list:
        v_GS_list, _ = calc_kernel_gradient_similarity(
            output_root=os.path.join(output_root, v),
            gt_root=os.path.join(gt_root, v)
        )
        GS_sum += sum(v_GS_list)
        kernel_num += len(v_GS_list)

        video_GS.append({
            'video_name': v,
            'gradient_similarity': v_GS_list
        })

    logs = []
    for v_lpips in video_GS:
        log = 'Video: {} Gradient-Similarity={:.4}'.format(
            v_lpips['video_name'],
            sum(v_lpips['gradient_similarity']) / len(v_lpips['gradient_similarity']))
        print(log)
        logs.append(log)
    log = 'Average LPIPS={:.4}'.format(GS_sum / kernel_num)
    print(log)
    logs.append(log)

    return video_GS, logs


def batch_calc_kernel_gradient_similarity(root_list, video_type=False):
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
            _, log = calc_kernel_gradient_similarity_video(ouput_root, gt_root)
        else:
            _, log = calc_kernel_gradient_similarity(ouput_root, gt_root)
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

import os
import scipy.io as scio
import torch
import cv2
import numpy as np
from base.os_base import handle_dir, get_fname_ext, listdir, glob_match


def save_flow_pt2mat(ori_root, dest_root):
    '''
    把 ori_root 中视频的 flow 从 pt 文件转换为 mat 文件后保存到 dest_root
    保存在 pt 文件中的 flow 的 shape=[1, 2, h, w]
    '''
    handle_dir(dest_root)
    video_list = listdir(ori_root)
    for v in video_list:
        handle_dir(os.path.join(dest_root, v))
        file_list = glob_match(os.path.join(ori_root, v, "*.pt"))
        for pt_path in file_list:
            fname, ext = get_fname_ext(pt_path)
            mat_path = os.path.join(dest_root, v, "{}.mat".format(fname))
            try:
                flow = torch.load(pt_path)[0].permute(1, 2, 0).cpu().numpy()
                flow_dict = {'flow': flow}
                scio.savemat(mat_path, flow_dict)
                print('save {} to {}'.format(pt_path, mat_path))
            except:
                print('skip file {}'.format(pt_path))


def visual_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    # flow shape: [h, w, 2]
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

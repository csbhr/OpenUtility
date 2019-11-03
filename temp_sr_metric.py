import os
import glob
import shutil
import cv2


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


def remove_video_postfix(root, postfix=''):
    video_list = sorted(os.listdir(root))
    for v in video_list:
        now_postfix = v.split("_")[-1]
        if now_postfix == postfix:
            src = os.path.join(root, v)
            dst = os.path.join(root, v[:-(len(postfix) + 1)])
            os.rename(src, dst)
            print('rename file from {} to {}'.format(src, dst))


def resort_frame_index(root, template='{:0>4}', start_list=None):
    '''
    重新排序并命名，以 template 为模板，以 start_list 的序号开始编号
    start_list:
        default: [0]
        如果 start_list 的长度与视频个数不一致，则所有视频都按 start_list[0] 开始编号
    '''
    template = template + '.{}'
    video_list = sorted(os.listdir(root))
    if not start_list:
        start_list = [0]
    if not len(start_list) == len(video_list):
        start_list = [start_list[0] for _ in range(len(video_list))]
    for start, v in zip(start_list, video_list):
        img_list = sorted(glob.glob(os.path.join(root, v, "*")))
        for i, im in enumerate(img_list):
            basename = os.path.basename(im)
            ext = basename.split(".")[-1]
            dest_basename = template.format(i + start, ext)
            src = im
            dst = os.path.join(root, v, dest_basename)
            os.rename(src, dst)
            print('rename file from {} to {}'.format(src, dst))


def rename_frame_by_index(root):
    video_list = sorted(os.listdir(root))
    for v in video_list:
        img_list = os.listdir(os.path.join(root, v))
        for im in img_list:
            idx = int(im.split("_")[0])
            src = os.path.join(root, v, im)
            dst = os.path.join(root, v, "{:0>4}.png".format(idx))
            os.rename(src, dst)
            print('rename file from {} to {}'.format(src, dst))


def remove_pre_tail_frames(root, recycle_bin='', num=1):
    '''
    删除各个视频的前后 num 帧
    删除的帧会复制到 recycle_bin 中
    '''
    video_list = os.listdir(root)
    if recycle_bin == '':
        recycle_bin = os.path.join(root, '_recycle_bin')
    handle_dir(recycle_bin)
    for v in video_list:
        img_list = sorted(glob.glob(os.path.join(root, v, "*")))
        handle_dir(os.path.join(recycle_bin, v))
        for i in range(num):
            src = img_list[i]
            dest = os.path.join(recycle_bin, v, os.path.basename(src))
            shutil.move(src, dest)
            print('remove file from {}'.format(src))
            src = img_list[-(i + 1)]
            dest = os.path.join(recycle_bin, v, os.path.basename(src))
            shutil.move(src, dest)
            print('remove file from {}'.format(src))


def corp_video_frames(ori_root, dest_root):
    handle_dir(dest_root)
    video_list = sorted(os.listdir(ori_root))
    for v in video_list:
        handle_dir(os.path.join(dest_root, v))
        file_list = glob.glob(os.path.join(ori_root, v, "*"))
        for ori_path in file_list:
            basename = os.path.basename(ori_path)
            dest_path = os.path.join(dest_root, v, basename)
            img = cv2.imread(ori_path)
            h, w, c = img.shape
            l_h, l_w = h // 4, w // 4
            new_h, new_w = (l_h - l_h % 8) * 4, (l_w - l_w % 8) * 4
            img = img[:new_h, :new_w, :]
            cv2.imwrite(dest_path, img)
            print("crop file", ori_path)

# # root = '/home/csbhr/workspace/sr_result/Real/RBPN'
# # root = '/home/csbhr/workspace/sr_result/REDS4/RBPN'
# # root = '/home/csbhr/workspace/sr_result/SPMC/RBPN'
# root = '/home/csbhr/workspace/sr_result/Vid4/RBPN'
# remove_video_postfix(root, postfix='4x')

# # root = '/home/csbhr/workspace/sr_result/Real/RBPN'
# # root = '/home/csbhr/workspace/sr_result/REDS4/RBPN'
# # root = '/home/csbhr/workspace/sr_result/SPMC/RBPN'
# root = '/home/csbhr/workspace/sr_result/Vid4/RBPN'
# rename_frame_by_index(root)

# root = '/home/csbhr/workspace/sr_result/SPMC/TOFlow'
# remove_pre_tail_frames(root, num=1)

# REDS4 start_list=[1]
# SPMC start_list=[1]
# Vid4 start_list=[1]
root = '/home/csbhr/workspace/sr_result/SPMC/TOFlow'
resort_frame_index(root, template='{:0>4}_toflow', start_list=[1])

# ori_root = '/home/csbhr/workspace/sr_result/Vid4/ours_deconv_finetune'
# dest_root = '/home/csbhr/workspace/sr_result/Vid4/ours_deconv_finetune_crop'
# corp_video_frames(ori_root, dest_root)

import shutil
import os
import glob
import torch
import scipy.io as scio


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


def VideoType2TypeVideo(ori_root, dest_root, ori_type, dest_type=''):
    '''
    videos/blur/frames.png  -->  blur/videos/frames.png
    '''
    if dest_type == '':
        dest_type = ori_type
    handle_dir(dest_root)
    handle_dir(os.path.join(dest_root, dest_type))
    video_list = os.listdir(ori_root)
    for v in video_list:
        image_list = os.listdir(os.path.join(ori_root, v, ori_type))
        handle_dir(os.path.join(dest_root, dest_type, v))
        for im in image_list:
            src = os.path.join(ori_root, v, ori_type, im)
            dst = os.path.join(dest_root, dest_type, v, im)
            shutil.copy(src, dst)
            print('copy file from {} to {}'.format(src, dst))


def TypeVideo2VideoType(ori_root, dest_root, ori_type, dest_type=''):
    '''
    blur/videos/frames.png  -->  videos/blur/frames.png
    '''
    if dest_type == '':
        dest_type = ori_type
    handle_dir(dest_root)
    video_list = os.listdir(os.path.join(ori_root, ori_type))
    for v in video_list:
        image_list = os.listdir(os.path.join(ori_root, ori_type, v))
        handle_dir(os.path.join(dest_root, v))
        handle_dir(os.path.join(dest_root, v, dest_type))
        for im in image_list:
            src = os.path.join(ori_root, ori_type, v, im)
            dst = os.path.join(dest_root, v, dest_type, im)
            shutil.copy(src, dst)
            print('copy file from {} to {}'.format(src, dst))


def extra_frames_from_videos(ori_root, dest_root, ori_postfix='', new_postfix='', ext='png'):
    '''
    从 ori_root 抽取后缀为 ori_postfix 的图片，
    修改后缀为 new_postfix 复制到 dest_root
    '''
    if new_postfix == '':
        new_postfix = ori_postfix
    handle_dir(dest_root)
    video_list = os.listdir(ori_root)
    for v in video_list:
        handle_dir(os.path.join(dest_root, v))
        flag_img_list = glob.glob(os.path.join(ori_root, v, "*_{}.{}".format(ori_postfix, ext)))
        for im in flag_img_list:
            video_ind = os.path.basename(im).split('_')[0]
            basename = '{}_{}.{}'.format(video_ind, new_postfix, ext)
            src = im
            dst = os.path.join(dest_root, v, basename)
            shutil.copy(src, dst)
            print('copy file from {} to {}'.format(src, dst))


def remove_frame_prefix(root, prefix=''):
    '''
    删除文件名的前缀
    '''
    video_list = os.listdir(root)
    for v in video_list:
        img_list = glob.glob(os.path.join(root, v, "*"))
        for im in img_list:
            basename = os.path.basename(im)
            now_prefix = basename[:len(prefix)]
            if now_prefix == prefix:
                dest_basename = basename[len(prefix):]
                src = im
                dst = os.path.join(root, v, dest_basename)
                os.rename(src, dst)
                print('rename file from {} to {}'.format(src, dst))


def remove_frame_postfix(root, postfix=''):
    '''
    删除文件名的后缀
    '''
    video_list = os.listdir(root)
    for v in video_list:
        img_list = glob.glob(os.path.join(root, v, "*"))
        for im in img_list:
            basename = os.path.basename(im)
            ext = basename.split('.')[-1]
            fname = basename[:-(len(ext) + 1)]
            now_postfix = fname[-len(postfix):]
            if now_postfix == postfix:
                dest_basename = "{}.{}".format(fname[:-len(postfix)], ext)
                src = im
                dst = os.path.join(root, v, dest_basename)
                os.rename(src, dst)
                print('rename file from {} to {}'.format(src, dst))


def add_frame_postfix(root, postfix=''):
    '''
    添加文件名的后缀
    '''
    video_list = os.listdir(root)
    for v in video_list:
        img_list = glob.glob(os.path.join(root, v, "*"))
        for im in img_list:
            basename = os.path.basename(im)
            filename = basename.split(".")[0]
            ext = basename.split(".")[-1]
            dest_basename = "{}_{}.{}".format(filename, postfix, ext)
            src = im
            dst = os.path.join(root, v, dest_basename)
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


def save_flow_pt2mat(ori_root, dest_root):
    '''
    把 ori_root 中的 flow 从 pt 文件转换为 mat 文件后保存到 dest_root
    '''
    handle_dir(dest_root)
    video_list = os.listdir(ori_root)
    for v in video_list:
        handle_dir(os.path.join(dest_root, v))
        file_list = glob.glob(os.path.join(ori_root, v, "*.pt"))
        for pt_path in file_list:
            filename = os.path.basename(pt_path)
            ext = filename.split('.')[-1]
            basename = filename[:-(len(ext) + 1)]
            mat_path = os.path.join(dest_root, v, "{}.mat".format(basename))
            try:
                flow = torch.load(pt_path)[0].permute(1, 2, 0).cpu().numpy()
                flow_dict = {'flow': flow}
                scio.savemat(mat_path, flow_dict)
                print('save {} to {}'.format(pt_path, mat_path))
            except:
                print('skip file {}'.format(pt_path))

import os
from base.os_base import handle_dir, copy_file, move_file, listdir, glob_match
from utils import file_regroup_utils


def VideoFlag2FlagVideo(ori_root, dest_root, ori_flag, dest_flag=None):
    '''
    videos/type/frames  -->  type/videos/frames
    params:
        ori_root: the dir of files that need to be processed
        dest_root: the dir for saving matched files
        ori_flag: the ori video flag(e.g. blur)
        dest_flag: the flag(e.g. blur) for saving videos
            default: None, that is keeping the ori flag
    '''
    if dest_flag is None:
        dest_flag = ori_flag
    handle_dir(dest_root)
    handle_dir(os.path.join(dest_root, dest_flag))
    video_list = listdir(ori_root)
    for v in video_list:
        image_list = listdir(os.path.join(ori_root, v, ori_flag))
        handle_dir(os.path.join(dest_root, dest_flag, v))
        for im in image_list:
            src = os.path.join(ori_root, v, ori_flag, im)
            dst = os.path.join(dest_root, dest_flag, v, im)
            copy_file(src, dst)


def FlagVideo2VideoFlag(ori_root, dest_root, ori_flag, dest_flag=None):
    '''
    blur/videos/frames  -->  videos/blur/frames
    params:
        ori_root: the dir of files that need to be processed
        dest_root: the dir for saving matched files
        ori_flag: the ori video flag(e.g. blur)
        dest_flag: the flag(e.g. blur) for saving videos
            default: None, that is keeping the ori flag
    '''
    if dest_flag is None:
        dest_flag = ori_flag
    handle_dir(dest_root)
    video_list = listdir(os.path.join(ori_root, ori_flag))
    for v in video_list:
        image_list = listdir(os.path.join(ori_root, ori_flag, v))
        handle_dir(os.path.join(dest_root, v))
        handle_dir(os.path.join(dest_root, v, dest_flag))
        for im in image_list:
            src = os.path.join(ori_root, ori_flag, v, im)
            dst = os.path.join(dest_root, v, dest_flag, im)
            copy_file(src, dst)


def remove_frames_prefix(root, prefix=''):
    '''
    remove prefix from frames
    params:
        root: the dir of videos that need to be processed
        prefix: the prefix to be removed
    '''
    video_list = listdir(root)
    for v in video_list:
        file_regroup_utils.remove_files_prefix(os.path.join(root, v), prefix=prefix)


def remove_frames_postfix(root, postfix=''):
    '''
    remove postfix from frames
    params:
        root: the dir of videos that need to be processed
        postfix: the postfix to be removed
    '''
    video_list = listdir(root)
    for v in video_list:
        file_regroup_utils.remove_files_postfix(os.path.join(root, v), postfix=postfix)


def add_frames_postfix(root, postfix=''):
    '''
    add postfix to frames
    params:
        root: the dir of videos that need to be processed
        postfix: the postfix to be added
    '''
    video_list = listdir(root)
    for v in video_list:
        file_regroup_utils.add_files_postfix(os.path.join(root, v), postfix=postfix)


def extra_frames_by_postfix(ori_root, dest_root, match_postfix='', new_postfix=None, match_ext='*'):
    '''
    extra frames from ori_root to dest_root by match_postfix and match_ext
    params:
        ori_root: the dir of videos that need to be processed
        dest_root: the dir for saving matched files
        match_postfix: the postfix to be matched
        new_postfix: the postfix for matched files
            default: None, that is keeping the ori postfix
        match_ext: the ext to be matched
    '''
    if new_postfix is None:
        new_postfix = match_postfix

    handle_dir(dest_root)
    video_list = listdir(ori_root)
    for v in video_list:
        file_regroup_utils.extra_files_by_postfix(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            match_postfix=match_postfix,
            new_postfix=new_postfix,
            match_ext=match_ext
        )


def resort_frames_index(root, template='{:0>4}', start_idx=0):
    '''
    resort frames' filename using template that index start from start_idx
    params:
        root: the dir of files that need to be processed
        template: the template for processed filename
        start_idx: the start index
    '''
    video_list = listdir(root)
    for v in video_list:
        file_regroup_utils.resort_files_index(os.path.join(root, v), template=template, start_idx=start_idx)


def remove_head_tail_frames(root, recycle_bin=None, num=0):
    '''
    remove num hean&tail frames from videos
    params:
        root: the dir of files that need to be processed
        recycle_bin: the removed frames will be put here
            defalut: None, that is putting the removed frames in root/_recycle_bin
        num: the number of frames to be removed
    '''
    if recycle_bin is None:
        recycle_bin = os.path.join(root, '_recycle_bin')
    handle_dir(recycle_bin)

    video_list = listdir(root)
    for v in video_list:
        img_list = sorted(glob_match(os.path.join(root, v, "*")))
        handle_dir(os.path.join(recycle_bin, v))
        for i in range(num):
            src = img_list[i]
            dest = os.path.join(recycle_bin, v, os.path.basename(src))
            move_file(src, dest)

            src = img_list[-(i + 1)]
            dest = os.path.join(recycle_bin, v, os.path.basename(src))
            move_file(src, dest)

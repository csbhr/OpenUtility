import os
from utils.base_utils import handle_dir
from utils.image_utils import batch_resize_images


def batch_resize_videos(ori_root, dest_root, scale=1.0, postfix=True):
    '''
    function:
        resizing videos in batches
    params:
        ori_root: the dir of videos that need to be processed
        dest_root: the dir to save processed videos
        scale: float, the resize scale
        postfix: if True: add scale as postfix for filename after resizing
    '''
    handle_dir(dest_root)
    videos = os.listdir(ori_root)
    for v in videos:
        ori_v = os.path.join(ori_root, v)
        dest_v = os.path.join(dest_root, v)
        batch_resize_images(ori_v, dest_v, scale=scale, postfix=postfix)
        print("video", v, "resize done !")


def ext_frames_from_videos(ori_root, save_root, fname_template='%4d.png', start_idx=0, end_idx=99999):
    handle_dir(save_root)
    videos = sorted(os.listdir(ori_root))[start_idx:end_idx]
    for v in videos:
        vn = v[:-(len(v.split('.')[-1]) + 1)]
        video_path = os.path.join(ori_root, v)
        png_dir = os.path.join(save_root, vn)
        png_path = os.path.join(png_dir, fname_template)
        handle_dir(png_dir)
        command = 'ffmpeg -i {} {}'.format(video_path, png_path)
        os.system(command)
        print("extra frames from {}".format(video_path))


def zip_frames_to_videos(ori_root, save_root, fname_template='%4d.png', video_ext='mp4', start_idx=0, end_idx=99999):
    handle_dir(save_root)
    videos_name = sorted(os.listdir(ori_root))[start_idx:end_idx]
    for vn in videos_name:
        imgs_path = os.path.join(ori_root, vn, fname_template)
        video_path = os.path.join(save_root, '{}.{}'.format(vn, video_ext))
        command = 'ffmpeg -r 24000/1001 -i {} -vcodec libx265 -pix_fmt yuv422p -crf 10 {}'.format(imgs_path, video_path)
        os.system(command)
        print("zip frames to {}".format(video_path))

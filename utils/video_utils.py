import os
from base.os_base import handle_dir, copy_file, listdir
from utils.image_utils import cv2_resize_images, matlab_resize_images, shift_images


def matlab_resize_videos(ori_root, dest_root, scale=1.0, method='bicubic', filename_template="{}.png"):
    '''
    function:
        resizing videos in batches, same as matlab2017 imresize
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        scale: float, the resize scale
        method: string, the interpolation method,
            optional: 'bilinear', 'bicubic'
            default: 'bicubic'
        filename_template: string, the filename template for saving images
    '''
    handle_dir(dest_root)
    videos = listdir(ori_root)
    for v in videos:
        matlab_resize_images(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            scale=scale,
            method=method,
            filename_template=filename_template
        )
        print("Video", v, "resize done !")


def cv2_resize_videos(ori_root, dest_root, scale=1.0, method='bicubic', filename_template="{}.png"):
    '''
    function:
        resizing videos in batches
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        scale: float, the resize scale
        method: string, the interpolation method,
            optional: 'nearest', 'bilinear', 'bicubic'
            default: 'bicubic'
        filename_template: string, the filename template for saving images
    '''
    handle_dir(dest_root)
    videos = listdir(ori_root)
    for v in videos:
        cv2_resize_images(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            scale=scale,
            method=method,
            filename_template=filename_template
        )
        print("Video", v, "resize done !")


def shift_videos(ori_root, dest_root, offset_x=0., offset_y=0., filename_template="{}.png"):
    '''
    function:
        shifting videos by (offset_x, offset_y) on (axis-x, axis-y) in batches
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        offset_x: float, offset pixels on axis-x
            positive=left; negative=right
        offset_y: float, offset pixels on axis-y
            positive=up; negative=down
        filename_template: string, the filename template for saving images
    '''
    handle_dir(dest_root)
    videos = listdir(ori_root)
    for v in videos:
        shift_images(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            offset_x=offset_x,
            offset_y=offset_y,
            filename_template=filename_template
        )
        print("Video", v, "shift done !")


def extra_frames_from_videos(ori_root, save_root, fname_template='%4d.png', ffmpeg_path='ffmpeg'):
    '''
    function:
        ext frames from videos
    params:
        ori_root: string, the dir of videos that need to be processed
        save_root: string, the dir to save processed frames
        fname_template: the template for frames' filename
        ffmpeg_path: ffmpeg path
    '''

    handle_dir(save_root)
    videos = sorted(listdir(ori_root))

    for v in videos:
        vn = v[:-(len(v.split('.')[-1]) + 1)]
        video_path = os.path.join(ori_root, v)
        png_dir = os.path.join(save_root, vn)
        png_path = os.path.join(png_dir, fname_template)
        handle_dir(png_dir)
        command = '{} -i {} {}'.format(ffmpeg_path, video_path, png_path)
        os.system(command)
        print("Extra frames from {}".format(video_path))


def zip_frames_to_videos(ori_root, save_root, fname_template='%4d.png', video_ext='mkv', ffmpeg_path='ffmpeg'):
    '''
    function:
        zip frames to videos
    params:
        ori_root: string, the dir of frames that need to be processed
        save_root: string, the dir to save processed videos
        fname_template: the template of frames' filename
        video_ext: the extension of videos
        ffmpeg_path: ffmpeg path
    '''

    handle_dir(save_root)
    videos_name = sorted(listdir(ori_root))

    for vn in videos_name:
        imgs_path = os.path.join(ori_root, vn, fname_template)
        video_path = os.path.join(save_root, '{}.{}'.format(vn, video_ext))
        command = '{} -i {} -c:v libx265 -x265-params lossless=1 {}'.format(
            ffmpeg_path, imgs_path, video_path
        )  # NTIRE 2022 Super-Resolution and Quality Enhancement of Compressed Video
        # command = '{} -r 24000/1001 -i {} -vcodec libx265 -pix_fmt yuv422p -crf 10 {}'.format(
        #     ffmpeg_path, imgs_path, video_path
        # )  # youku competition
        os.system(command)
        print("Zip frames to {}".format(video_path))


def copy_frames_for_fps(ori_root, save_root, mul=12, fname_template="{:0>4}", ext="png"):
    '''
    function:
        copy frames for fps
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        mul: the multiple of copy
        fname_template: the template of frames' filename
        ext: the ext of frames' filename
    '''
    fname_template = fname_template + '.{}'
    videos_name = sorted(listdir(ori_root))
    handle_dir(save_root)
    for vn in videos_name:
        frmames = sorted(listdir(os.path.join(ori_root, vn)))
        handle_dir(os.path.join(save_root, vn))
        for i, f in enumerate(frmames):
            for j in range(mul):
                now_idx = i * mul + j
                src = os.path.join(ori_root, vn, f)
                dest = os.path.join(save_root, vn, fname_template.format(now_idx, ext))
                copy_file(src, dest)

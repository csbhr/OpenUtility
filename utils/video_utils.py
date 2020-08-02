import os
from base.os_base import handle_dir, copy_file, listdir
from utils.image_utils import batch_cv2_resize_images, batch_matlab_resize_images


def batch_matlab_resize_videos(ori_root, dest_root, scale=1.0, method='bicubic', filename_template="{}.png"):
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
        batch_matlab_resize_images(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            scale=scale,
            method=method,
            filename_template=filename_template
        )
        print("Video", v, "resize done !")


def batch_cv2_resize_videos(ori_root, dest_root, scale=1.0, method='bicubic', filename_template="{}.png"):
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
        batch_cv2_resize_images(
            ori_root=os.path.join(ori_root, v),
            dest_root=os.path.join(dest_root, v),
            scale=scale,
            method=method,
            filename_template=filename_template
        )
        print("Video", v, "resize done !")


def extra_frames_from_videos(ori_root, save_root, fname_template='%4d.png', start_end=None):
    '''
    function:
        ext frames from videos
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        fname_template: the template for frames' filename
        start_end: list, len=2, the start and end index for processed videos,
            assert: len(start_end)=2
            default: None, that is processing all videos
    '''

    handle_dir(save_root)

    videos = sorted(listdir(ori_root))
    if start_end is not None:
        assert len(start_end) == 2, "only support len(start_end)=2"
        videos = videos[start_end[0]:start_end[1]]

    for v in videos:
        vn = v[:-(len(v.split('.')[-1]) + 1)]
        video_path = os.path.join(ori_root, v)
        png_dir = os.path.join(save_root, vn)
        png_path = os.path.join(png_dir, fname_template)
        handle_dir(png_dir)
        command = 'ffmpeg -i {} {}'.format(video_path, png_path)
        os.system(command)
        print("Extra frames from {}".format(video_path))


def zip_frames_to_videos(ori_root, save_root, fname_template='%4d.png', video_ext='mp4', start_end=None):
    '''
    function:
        zip frames to videos
    params:
        ori_root: string, the dir of videos that need to be processed
        dest_root: string, the dir to save processed videos
        fname_template: the template of frames' filename
        start_end: list, len=2, the start and end index for processed videos,
            assert: len(start_end)=2
            default: None, that is processing all videos
    '''
    handle_dir(save_root)

    videos_name = sorted(listdir(ori_root))
    if start_end is not None:
        assert len(start_end) == 2, "only support len(start_end)=2"
        videos_name = videos_name[start_end[0]:start_end[1]]

    for vn in videos_name:
        imgs_path = os.path.join(ori_root, vn, fname_template)
        video_path = os.path.join(save_root, '{}.{}'.format(vn, video_ext))
        command = 'ffmpeg -i {} -vcodec libx264 -crf 16 -pix_fmt yuv420p {}'.format(imgs_path, video_path)
        # command = 'ffmpeg -r 24000/1001 -i {} -vcodec libx265 -pix_fmt yuv422p -crf 10 {}'.format(
        #     imgs_path, video_path)  # youku competition
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

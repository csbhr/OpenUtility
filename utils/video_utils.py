import os


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


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

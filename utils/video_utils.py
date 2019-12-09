import os
import glob
import cv2


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


def resize_videos(ori_root, save_root, scale):
    handle_dir(save_root)
    videos_path = sorted(glob.glob(os.path.join(ori_root, "*")))
    for vp in videos_path:
        video_name = os.path.basename(vp)
        imgs_path = sorted(glob.glob(os.path.join(vp, "*")))
        for ip in imgs_path:
            img_name = os.path.basename(ip)
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            h, w, c = img.shape
            bicubic_img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)

            savepath = os.path.join(save_root, video_name)
            handle_dir(savepath)
            cv2.imwrite(os.path.join(savepath, img_name), bicubic_img)
            print('resize img:', ip)


# ori_root = '/media/csbhr/1_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/video/SDR_540p'
# save_root = '/media/csbhr/2_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/png/LR'
# ext_frames_from_videos(ori_root, save_root, start_idx=300, end_idx=700)
#
# ori_root = '/media/csbhr/1_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/video/SDR_4K'
# save_root = '/media/csbhr/2_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/png/GT'
# ext_frames_from_videos(ori_root, save_root, start_idx=300, end_idx=700)

# ori_root = '/media/csbhr/1_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/初赛/测试集/SDR_540p'
# save_root = '/media/csbhr/2_Disk2T/BHR/Dataset/SR_DataSet/AI+4K_HDR/test_first/LR'
# ext_frames_from_videos(ori_root, save_root)

ori_root = '/media/csbhr/2_Disk2T/BHR/Dataset/SR_DataSet/REDS/train_240_269/HR'
save_root = '/media/csbhr/2_Disk2T/BHR/Dataset/SR_DataSet/REDS/train_240_269/LR'
resize_videos(ori_root, save_root, scale=0.25)

# ori_root = '/media/csbhr/1_Disk2T/temp/test_first/results/sharp_bicubic/png3'
# save_root = '/media/csbhr/2_Disk2T/temp/results/sharp_bicubic/video2'
# zip_frames_to_videos(ori_root, save_root)

import os
import numpy as np
import imageio


frame_root_1 = '/home/csbhr/Downloads/ours'
frame_root_2 = '/home/csbhr/Downloads/input'
gif_path = '/home/csbhr/Downloads/visual.gif'

frames_fn_1 = sorted(os.listdir(frame_root_1))
frames_fn_2 = sorted(os.listdir(frame_root_2))

tmp_img = imageio.imread(os.path.join(frame_root_1, frames_fn_1[0]))
H, W, C = tmp_img.shape
gif = np.zeros(tmp_img.shape, dtype=tmp_img.dtype)
line = np.zeros(tmp_img.shape, dtype=tmp_img.dtype)
line[:, :, 0] = 0
line[:, :, 1] = 255
line[:, :, 2] = 0

nl = 30
wl = 3
gif_images = []
for i, (fn1, fn2) in enumerate(zip(frames_fn_1, frames_fn_2)):
    frame_1 = imageio.imread(os.path.join(frame_root_1, fn1))
    gif = frame_1
    if i < nl:
        frame_2 = imageio.imread(os.path.join(frame_root_2, fn2))
        gif[:, :W-(W // nl * i), :] = frame_2[:, :W-(W // nl * i), :]
        gif[:, W-(W // nl * i + wl):W-(W // nl * i), :] = line[:, W-(W // nl * i + wl):W-(W // nl * i), :]
    gif_images.append(gif.copy())
imageio.mimsave(gif_path, gif_images, fps=4)


# frames_root = '/home/csbhr/Downloads/(TPAMI) Self-Supervised Deep Blind Video Super-Resolution Supplemental Material/figures/videos'
# gif_root = '/home/csbhr/Downloads/(TPAMI) Self-Supervised Deep Blind Video Super-Resolution Supplemental Material/figures'
#
# methods = sorted(os.listdir(frames_root))
# for me in methods:
#     frames_fn = sorted(os.listdir(os.path.join(frames_root, me)))
#     gif_images = []
#     for fn in frames_fn:
#         gif_images.append(imageio.imread(os.path.join(frames_root, me, fn)))
#     imageio.mimsave(os.path.join(gif_root, "{}.gif".format(me)), gif_images, fps=2)

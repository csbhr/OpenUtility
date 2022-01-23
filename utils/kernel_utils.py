import os
import cv2
from base.os_base import handle_dir, get_fname_ext, listdir
from base.kernel_base import kernel2png, load_mat_kernel


def save_kernels_as_png(ori_root, dest_root, filename_template="{}.png"):
    '''
    function:
        convert kernel for saving as png
    params:
        ori_root: string, the dir of kernels that need to be processed
        dest_root: string, the dir to save processed kernels
        filename_template: string, the filename template for saving kernels
    '''

    handle_dir(dest_root)
    kernels_fname = sorted(listdir(ori_root))
    for kerf in kernels_fname:
        ker = load_mat_kernel(os.path.join(ori_root, kerf))
        ker_png = kernel2png(ker)
        cv2.imwrite(os.path.join(dest_root, filename_template.format(get_fname_ext(kerf)[0])), ker_png)
        print("Kernel", kerf, "save done !")

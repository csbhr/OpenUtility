import shutil
import os
import glob
from utils.base_utils import handle_dir


def extra_imgs_by_postfix(ori_root, dest_root, ori_postfix='', new_postfix='', ext='*'):
    '''
    从 ori_root 抽取后缀为 ori_postfix 的图片，
    修改后缀为 new_postfix 复制到 dest_root
    '''
    if new_postfix == '':
        new_postfix = ori_postfix
    handle_dir(dest_root)
    flag_img_list = glob.glob(os.path.join(ori_root, "*_{}.{}".format(ori_postfix, ext)))
    for im in flag_img_list:
        basename = os.path.basename(im)
        indx = basename.split('_')[0]
        this_ext = basename.split('.')[-1]
        filename = '{}_{}.{}'.format(indx, new_postfix, this_ext)
        src = im
        dst = os.path.join(dest_root, filename)
        shutil.copy(src, dst)
        print('copy file from {} to {}'.format(src, dst))


def remove_imgs_prefix(root, prefix=''):
    '''
    删除文件名的前缀
    '''
    img_list = glob.glob(os.path.join(root, "*"))
    for im in img_list:
        basename = os.path.basename(im)
        now_prefix = basename[:len(prefix)]
        if now_prefix == prefix:
            dest_basename = basename[len(prefix):]
            src = im
            dst = os.path.join(root, dest_basename)
            os.rename(src, dst)
            print('rename file from {} to {}'.format(src, dst))


def remove_imgs_postfix(root, postfix=''):
    '''
    删除文件名的后缀
    '''
    img_list = glob.glob(os.path.join(root, "*"))
    for im in img_list:
        basename = os.path.basename(im)
        ext = basename.split('.')[-1]
        fname = basename[:-(len(ext) + 1)]
        now_postfix = fname[-len(postfix):]
        if now_postfix == postfix:
            dest_basename = "{}.{}".format(fname[:-len(postfix)], ext)
            src = im
            dst = os.path.join(root, dest_basename)
            os.rename(src, dst)
            print('rename file from {} to {}'.format(src, dst))


def add_imgs_postfix(root, postfix=''):
    '''
    添加文件名的后缀
    '''
    img_list = glob.glob(os.path.join(root, "*"))
    for im in img_list:
        basename = os.path.basename(im)
        ext = basename.split(".")[-1]
        filename = basename[:-(len(ext) + 1)]
        dest_basename = "{}_{}.{}".format(filename, postfix, ext)
        src = im
        dst = os.path.join(root, dest_basename)
        os.rename(src, dst)
        print('rename file from {} to {}'.format(src, dst))


def resort_imgs_index(root, template='{:0>4}', start=0):
    '''
    重新排序并命名，以 template 为模板，以 start 的序号开始编号
    '''
    template = template + '.{}'
    img_list = sorted(glob.glob(os.path.join(root, "*")))
    for i, im in enumerate(img_list):
        basename = os.path.basename(im)
        ext = basename.split(".")[-1]
        dest_basename = template.format(i + start, ext)
        src = im
        dst = os.path.join(root, dest_basename)
        os.rename(src, dst)
        print('rename file from {} to {}'.format(src, dst))

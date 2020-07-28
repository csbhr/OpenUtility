import os
from base.os_base import handle_dir, get_fname_ext, copy_file, rename_file, glob_match


def remove_files_prefix(root, prefix=''):
    '''
    remove prefix from filename
    params:
        root: the dir of files that need to be processed
        prefix: the prefix to be removed
    '''
    img_list = glob_match(os.path.join(root, "*"))
    for im in img_list:
        basename = os.path.basename(im)
        now_prefix = basename[:len(prefix)]
        if now_prefix == prefix:
            dest_basename = basename[len(prefix):]
            src = im
            dst = os.path.join(root, dest_basename)
            rename_file(src, dst)


def remove_files_postfix(root, postfix=''):
    '''
    remove postfix from filename
    params:
        root: the dir of files that need to be processed
        postfix: the postfix to be removed
    '''
    img_list = glob_match(os.path.join(root, "*"))
    for im in img_list:
        fname, ext = get_fname_ext(im)
        now_postfix = fname[-len(postfix):]
        if now_postfix == postfix:
            dest_basename = "{}.{}".format(fname[:-len(postfix)], ext)
            src = im
            dst = os.path.join(root, dest_basename)
            rename_file(src, dst)


def add_files_postfix(root, postfix=''):
    '''
    add postfix to filename
    params:
        root: the dir of files that need to be processed
        postfix: the postfix to be added
    '''
    img_list = glob_match(os.path.join(root, "*"))
    for im in img_list:
        fname, ext = get_fname_ext(im)
        dest_basename = "{}{}.{}".format(fname, postfix, ext)
        src = im
        dst = os.path.join(root, dest_basename)
        rename_file(src, dst)


def extra_files_by_postfix(ori_root, dest_root, match_postfix='', new_postfix=None, match_ext='*'):
    '''
    extra files from ori_root to dest_root by match_postfix and match_ext
    params:
        ori_root: the dir of files that need to be processed
        dest_root: the dir for saving matched files
        match_postfix: the postfix to be matched
        new_postfix: the postfix for matched files
            default: None, that is keeping the ori postfix
        match_ext: the ext to be matched
    '''
    if new_postfix is None:
        new_postfix = match_postfix

    handle_dir(dest_root)
    flag_img_list = glob_match(os.path.join(ori_root, "*{}.{}".format(match_postfix, match_ext)))
    for im in flag_img_list:
        fname, ext = get_fname_ext(im)
        dest_basename = '{}{}.{}'.format(fname[-len(match_postfix):], new_postfix, ext)
        src = im
        dst = os.path.join(dest_root, dest_basename)
        copy_file(src, dst)


def resort_files_index(root, template='{:0>4}', start_idx=0):
    '''
    resort files' filename using template that index start from start_idx
    params:
        root: the dir of files that need to be processed
        template: the template for processed filename
        start_idx: the start index
    '''
    template = template + '.{}'
    img_list = sorted(glob_match(os.path.join(root, "*")))
    for i, im in enumerate(img_list):
        fname, ext = get_fname_ext(im)
        dest_basename = template.format(i + start_idx, ext)
        src = im
        dst = os.path.join(root, dest_basename)
        rename_file(src, dst)

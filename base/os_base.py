import shutil
import os
import glob


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('mkdir:', dir)


def listdir(path):
    sys_files = ['.DS_Store']
    files = os.listdir(path)
    for sf in sys_files:
        if sf in files:
            files.remove(sf)
    return files


def glob_match(template):
    fpathes = glob.glob(template)
    dest_fpathes = []
    for fp in fpathes:
        if '.DS_Store' not in fp:
            dest_fpathes.append(fp)
    return dest_fpathes


def get_fname_ext(filepath):
    filename = os.path.basename(filepath)
    ext = filename.split(".")[-1]
    fname = filename[:-(len(ext) + 1)]
    return fname, ext


def copy_file(src, dst):
    shutil.copy(src, dst)
    print('copy file from {} to {}'.format(os.path.basename(src), os.path.basename(dst)))


def move_file(src, dst):
    shutil.move(src, dst)
    print('move file from {} to {}'.format(os.path.basename(src), os.path.basename(dst)))


def rename_file(src, dst):
    os.rename(src, dst)
    print('rename file from {} to {}'.format(os.path.basename(src), os.path.basename(dst)))

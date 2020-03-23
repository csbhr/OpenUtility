# Python_Tools
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/Python_Tools/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

There are some useful tools for low-level vision tasks.

- [Calculate images' PSNR/SSIM](#chapter-1)
- [Calculate videos' PSNR/SSIM](#chapter-2)
- [Calculate the properties of model](#chapter-3)
- [Crop and combine images](#chapter-4)


<a name="chapter-1"></a>
### Calculate images' PSNR/SSIM
- You can calculate the PSNR/SSIM of the images in batches according to the following demo:
```
from utils.calc_image_psnr_ssim import batch_calc_image_PSNR_SSIM
root_list = [
    {
        'output': '/path/to/output images 1',
        'gt': '/path/to/gt images 1'
    },
    {
        'output': '/path/to/output images 2',
        'gt': '/path/to/gt images 2'
    },
]
log_list = batch_calc_image_PSNR_SSIM(root_list)
print("--------------------------------------------------------------------------------------")
for i, log in enumerate(log_list):
    print(">> The {}-th:".format(i), log['data_path'])
    print(log['log'])
```

<a name="chapter-2"></a>
### Calculate videos' PSNR/SSIM
- You can calculate the PSNR/SSIM of the videos in batches according to the following demo:
```
from utils.calc_video_psnr_ssim import batch_calc_video_PSNR_SSIM_toCSV
root_list = [
    {
        'output': '/path/to/output videos 1',
        'gt': '/path/to/gt videos 1'
    },
    {
        'output': '/path/to/output videos 2',
        'gt': '/path/to/gt videos 2'
    },
]
save_csv_log_root = '/path/to/log'
batch_calc_video_PSNR_SSIM_toCSV(root_list, save_csv_log_root)
```
   
<a name="chapter-3"></a>
### Calculate the properties of model
- We use Swall0w's tools [torchstat](https://github.com/Swall0w/torchstat)
- You should run the commands to install torchstat:
```
cd ./utils/torchstat
python setup.py install
```
- The properties of model include: Params、Memory、MAdd、Flops、MemR+W
- You can calculate properties of model according to the following demo:
```
from torchstat import stat
network = None  # Please define the model
input_size = (3, 80, 80)  # the size of input (channel, height, width)
stat(network, input_size)
```
   
<a name="chapter-4"></a>
### Crop and combine images
- When you need to infer large image, you can crop image to many patches with padding according to the following demo:
```
# Notice: 
#   filenames should not contain the character "-"
#   the crop flag "x-x-x-x" will be at the end of filename when cropping
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_crop_img_with_padding(ori_root, dest_root, min_size=(800, 800), padding=100)
```
- When you finish inferring large image with cropped patches, you can combine patches to image according to the following demo:
```
# Notice: 
#   filenames should not contain the character "-" except for the crop flag
#   the crop flag "x-x-x-x" should be at the end of filename when combining
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_combine_img(ori_root, dest_root, padding=100)
```
- You can traversal crop image to many patches with same interval according to the following demo:
```
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_traverse_crop_img(ori_root, dest_root, dsize=(800, 800), interval=400)
```
# OpenUtility
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/Python_Tools/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

There are some useful tools for low-level vision tasks.

- [Calculate images' PSNR/SSIM](#chapter-1)
- [Calculate videos' PSNR/SSIM](#chapter-2)
- [Calculate the properties of model](#chapter-3)
- [Crop and combine images](#chapter-4)
- [Operate csv file](#chapter-5)
- [Plot multiple curves in one figure](#chapter-6)
- [Visualize optical flow](#chapter-7)


## Dependencies

- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
- pandas: `conda install pandas`

## User Guide
Here are some simple demos. If you want to learn more about the usage of these tools, you can refer to the optional parameters of functions in the source file.


<a name="chapter-1"></a>
### Calculate images' PSNR/SSIM
- You can calculate the PSNR/SSIM of the images in batches by following the demo:
```python
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
- You can calculate the PSNR/SSIM of the videos in batches by following the demo:
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
- You can calculate properties of model by following the demo:
```
from torchstat import stat
network = None  # Please define the model
input_size = (3, 80, 80)  # the size of input (channel, height, width)
stat(network, input_size)
```
   
<a name="chapter-4"></a>
### Crop and combine images
- When you need to infer large image, you can crop image to many patches with padding by following the demo:
```
# Notice: 
#   filenames should not contain the character "-"
#   the crop flag "x-x-x-x" will be at the end of filename when cropping
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_crop_img_with_padding(ori_root, dest_root, min_size=(800, 800), padding=100)
```
- When you finish inferring large image with cropped patches, you can combine patches to image by following the demo:
```
# Notice: 
#   filenames should not contain the character "-" except for the crop flag
#   the crop flag "x-x-x-x" should be at the end of filename when combining
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_combine_img(ori_root, dest_root, padding=100)
```
- You can traversal crop image to many patches with same interval by following the demo:
```
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_traverse_crop_img(ori_root, dest_root, dsize=(800, 800), interval=400)
```

<a name="chapter-5"></a>
### Operate csv file
- You can read a csv file by following the demo:
```
from utils import file_io_utils
data, col_names, row_names = file_io_utils.read_csv('filename.csv', col_name_ind=0, row_name_ind=0)
```
- You can write a numpy.array into a csv file by following the demo:
```
from utils import file_io_utils
row_names = ['r1', 'r2', 'r3']
col_names = ['c1', 'c2', 'c3', 'c4']
data_array = np.array([[1, 2, 3, 4],
                       [2, 3, 4, 5],
                       [3, 4, 5, 6]])
file_io_utils.write_csv('filename.csv', data_array, col_names, row_names)
```

<a name="chapter-6"></a>
### Plot multiple curves in one figure
- You plot multiple curves in one figure by following the demo:
```
import numpy as np
from utils import visual_utils
array_list = [
    np.array([1, 4, 5, 3, 6]),
    np.array([2, 3, 7, 4, 5]),
]
label_list = ['curve-1', 'curve-2']
visual_utils.plot_multi_curve(array_list, label_list)
```

<a name="chapter-7"></a>
### Visualize optical flow
- You can visualize optical flow by following the demo:
```
from utils import visual_utils
flow = None  # this is flow, shape=(h, w, 2)
rgb_image = visual_utils.visual_flow(flow)
```

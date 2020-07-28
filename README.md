# OpenUtility
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/Python_Tools/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

There are some useful tools for low-level vision tasks.

- [Calculate images' PSNR/SSIM](#chapter-1)
- [Calculate videos' PSNR/SSIM](#chapter-2)
- [Resize images/videos](#chapter-3)
- [Crop, combine and traverse images](#chapter-4)
- [Calculate the properties of deep model: Params, Flops, etc.](#chapter-5)
- [Operate csv file](#chapter-6)
- [Plot multiple curves in one figure](#chapter-7)
- [Visualize optical flow](#chapter-8)


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
from utils.image_metric_utils import batch_calc_image_PSNR_SSIM
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
batch_calc_image_PSNR_SSIM(root_list)
```

<a name="chapter-2"></a>
### Calculate videos' PSNR/SSIM
- You can calculate the PSNR/SSIM of the videos in batches by following the demo:
```python
from utils.video_metric_utils import batch_calc_video_PSNR_SSIM
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
batch_calc_video_PSNR_SSIM(root_list)
```

<a name="chapter-3"></a>
### Resize images/videos
- You can resize images using cv2 in batches by following the demo:
```python
from utils.image_utils import batch_cv2_resize_images
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_cv2_resize_images(ori_root, dest_root, scale=2.0)
```
- You can resize videos in batches by following the demo:
```python
from utils.video_utils import batch_cv2_resize_videos
ori_root = '/path/to/ori videos'
dest_root = '/path/to/dest videos'
batch_cv2_resize_videos(ori_root, dest_root, scale=2.0)
```
   
<a name="chapter-4"></a>
### Crop and combine images
- When you need to infer large image, you can crop image to many patches with padding by following the demo:
```python
# Notice: 
#   filenames should not contain the character "-"
#   the crop flag "x-x-x-x" will be at the end of filename when cropping
from utils.image_crop_combine_utils import *
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_crop_img_with_padding(ori_root, dest_root, min_size=(800, 800), padding=100)
```
- When you finish inferring large image with cropped patches, you can combine patches to image by following the demo:
```python
# Notice: 
#   filenames should not contain the character "-" except for the crop flag
#   the crop flag "x-x-x-x" should be at the end of filename when combining
from utils.image_crop_combine_utils import *
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_combine_img(ori_root, dest_root, padding=100)
```
- You can traversal crop image to many patches with same interval by following the demo:
```python
from utils.image_crop_combine_utils import *
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_traverse_crop_img(ori_root, dest_root, dsize=(800, 800), interval=400)
```
- You can select valid patch that are not too smooth by following the demo:
```python
from utils.image_crop_combine_utils import *
ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
batch_select_valid_patch(ori_root, dest_root)
```
   
<a name="chapter-5"></a>
### Calculate the properties of deep model: Params, Flops, etc.
- We use Swall0w's tools [torchstat](https://github.com/Swall0w/torchstat)
- You should run the commands to install torchstat:
```
cd ./torchstat
python setup.py install
```
- The properties of model include: Params、Memory、MAdd、Flops、MemR+W
- You can calculate properties of model by following the demo:
```python
from torchstat import stat
network = None  # Please define the model
input_size = (3, 80, 80)  # the size of input (channel, height, width)
stat(network, input_size)
```

<a name="chapter-6"></a>
### Operate csv file
- You can read a csv file by following the demo:
```python
from base import file_io_base
data, col_names, row_names = file_io_base.read_csv('filename.csv', col_name_ind=0, row_name_ind=0)
```
- You can write a numpy.array into a csv file by following the demo:
```python
from base import file_io_base
import numpy as np
row_names = ['r1', 'r2', 'r3']
col_names = ['c1', 'c2', 'c3', 'c4']
data_array = np.array([[1, 2, 3, 4],
                       [2, 3, 4, 5],
                       [3, 4, 5, 6]])
file_io_base.write_csv('filename.csv', data_array, col_names, row_names)
```

<a name="chapter-7"></a>
### Plot multiple curves in one figure
- You plot multiple curves in one figure by following the demo:
```python
import numpy as np
from utils import plot_utils
array_list = [
    np.array([1, 4, 5, 3, 6]),
    np.array([2, 3, 7, 4, 5]),
]
label_list = ['curve-1', 'curve-2']
plot_utils.plot_multi_curve(array_list, label_list)
```

<a name="chapter-8"></a>
### Visualize optical flow
- You can visualize optical flow by following the demo:
```python
from utils import flow_utils
flow = None  # this is flow, shape=(h, w, 2)
rgb_image = flow_utils.visual_flow(flow)
```

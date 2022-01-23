# OpenUtility
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/Python_Tools/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

There are some useful tools for low-level vision tasks.

- [Calculating Metrics ( PSNR, SSIM, etc.)](#chapter-calculating-metrics)
- [Image / Video Processing ( resize, crop, shift, etc.)](#chapter-image-video-processing)
- [Deep Model Properties ( Params, Flops, etc. )](#chapter-model-properties)
- [File Processing ( csv, etc. )](#chapter-file-processing)
- [Visualize Tools ( plot, optical-flow, etc. )](#chapter-visualize-tools)


## Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch](https://pytorch.org/)
- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
- pandas: `conda install pandas`

## Easy to use
Here are some simple demos. If you want to learn more about the usage of these tools, you can refer to the source code for more optional parameters.


<a name="chapter-calculating-metrics"></a>
### 1. Calculating Metrics ( PSNR, SSIM, etc.)

##### 1.1 PSNR/SSIM
- Following the demo for batch operation:
```python
# Images
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

# Videos
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

##### 1.2 LPIPS
- Following the demo for batch operation:
```python
# Images
from utils.image_metric_utils import batch_calc_image_LPIPS
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
batch_calc_image_LPIPS(root_list)

# Videos
from utils.video_metric_utils import batch_calc_video_LPIPS
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
batch_calc_video_LPIPS(root_list)
```

##### 1.3 NIQE
- Following the demo for batch operation:
```python
# Images
from utils.image_metric_utils import batch_calc_image_NIQE
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
batch_calc_image_NIQE(root_list)

# Videos
from utils.video_metric_utils import batch_calc_video_NIQE
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
batch_calc_video_NIQE(root_list)
```

##### 1.4 Kernel Gradient Similarity
- Following the demo for batch operation:
```python
# Images: video_type=False
# Videos: video_type=True
from utils.kernel_metric_utils import batch_calc_kernel_metric
root_list = [
    {
        'output': '/path/to/output kernel 1',
        'gt': '/path/to/gt kernel 1'
    },
    {
        'output': '/path/to/output kernel 2',
        'gt': '/path/to/gt kernel 2'
    },
]
batch_calc_kernel_metric(root_list, video_type=False)
```


<a name="chapter-image-video-processing"></a>
### 2. Image / Video Processing ( resize, crop, shift, etc.)

##### 2.1 Image/Video Resize
- We use fatheral's python implementation of matLab imresize() function [fatheral/matlab_imresize](https://github.com/fatheral/matlab_imresize).
- Following the demo for batch operation:

```python
# Images
from utils.image_utils import matlab_resize_images

ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
matlab_resize_images(ori_root, dest_root, scale=2.0)

# Videos
from utils.video_utils import matlab_resize_videos

ori_root = '/path/to/ori videos'
dest_root = '/path/to/dest videos'
matlab_resize_videos(ori_root, dest_root, scale=2.0)
```
- We also apply opencv for resizing.
- Following the demo for batch operation:

```python
from utils.image_utils import cv2_resize_images

ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
cv2_resize_images(ori_root, dest_root, scale=2.0)

# Videos
from utils.video_utils import cv2_resize_videos

ori_root = '/path/to/ori videos'
dest_root = '/path/to/dest videos'
cv2_resize_videos(ori_root, dest_root, scale=2.0)
```

##### 2.2 Crop and combine images
- When you need to infer large image, you can crop image to many patches with padding by following the demo:
```python
# Notice: 
#   filenames should not contain the character "-"
#   the crop flag "x-x-x-x" will be at the end of filename when cropping
#   the combine operation will use the crop flag "x-x-x-x"
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

##### 2.3 Image/Video Shift
- We use "Bilinear" interpolation method to shift images/videos for sub-pixels.
- Following the demo for batch operation:

```python
# Images
from utils.image_utils import shift_images

ori_root = '/path/to/ori images'
dest_root = '/path/to/dest images'
shift_images(ori_root, dest_root, offset_x=0.5, offset_y=0.5)

# Videos
from utils.video_utils import shift_videos

ori_root = '/path/to/ori videos'
dest_root = '/path/to/dest videos'
shift_videos(ori_root, dest_root, offset_x=0.5, offset_y=0.5)
```


<a name="chapter-model-properties"></a>
### 3. Deep Model Properties ( Params, Flops, etc. )
- You can only calculate model Params by following the demo:
```python
from utils.dnn_utils import cal_parmeters
network = None  # Please define the model
cal_parmeters(network)
```
- You can also calculate more properties of model: Params, Memory, MAdd, Flops, etc.
- We use Swall0w's tools [Swall0w/torchstat](https://github.com/Swall0w/torchstat).
- [Swall0w/torchstat] can not use cuda, so we modified it for using cuda. If you want to calculate Flops on cuda, please using the following command to install torchstat.
```shell script
cd ./utils/torchstat
python3 setup.py install
```
- If you do not want to using cuda, please using the following command to install torchstat.
```shell script
pip install torchstat  # pytorch >= 1.0.0
pip install torchstat==0.0.6  # pytorch = 0.4.1
```
- And then you can calculate properties by following the demo:
```python
from torchstat import stat
network = None  # Please define the model
input_size = (3, 80, 80)  # the size of input (channel, height, width)
stat(network, input_size)
```


<a name="chapter-file-processing"></a>
### 4. File Processing ( csv, etc. )

##### 4.1 csv file
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


<a name="chapter-visualize-tools"></a>
### 5. Visualize Tools ( plot, optical-flow, etc. )

##### 5.1 Plot multiple curves in one figure
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

##### 5.2 Visualize optical flow
- You can visualize optical flow by following the demo:
```python
from utils import flow_utils
flow = None  # this is flow, shape=(h, w, 2)
rgb_image = flow_utils.visual_flow(flow)
```

# Python_Tools
There are some useful tools for low-level vision tasks.

## Current released function:

#### Calculate the PSNR/SSIM of images
- Follow the demo in demo_image_psnr_ssim.py
   
#### Calculate the PSNR/SSIM of videos
- Follow the demo in demo_video_psnr_ssim.py
   
#### Calculate the properties of model
- We use Swall0w's tools [torchstat](https://github.com/Swall0w/torchstat)
- You should run the commands to install torchstat:
```
    cd ./utils/torchstat
    python setup.py install
```
- Follow the demo in demo_flops_params.py
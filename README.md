# Python_Tools
There are some useful tools for low-level vision tasks.

- [Calculate the PSNR/SSIM of images](#chapter-1)
- [Calculate the PSNR/SSIM of videos](#chapter-2)
- [Calculate the properties of model](#chapter-3)

<a name="chapter-1"></a>
### Calculate the PSNR/SSIM of images
- Follow the demo in demo_image_psnr_ssim.py

<a name="chapter-2"></a>
### Calculate the PSNR/SSIM of videos
- Follow the demo in demo_video_psnr_ssim.py
   
<a name="chapter-3"></a>
### Calculate the properties of model
- We use Swall0w's tools [torchstat](https://github.com/Swall0w/torchstat)
- You should run the commands to install torchstat:
```
    cd ./utils/torchstat
    python setup.py install
```
- Follow the demo in demo_flops_params.py
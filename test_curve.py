from utils.visual_utils import plot_multi_curve
import torch

label_list = [
    'experiment_flag',
]
experiment_list = [
    'experiment_dirname',
]

max_epoch = 9999

loss_array_list = []
psnr_array_list = []
for i in range(len(experiment_list)):
    log_dir = '../experiment/{}'.format(experiment_list[i])
    loss_log = torch.load(log_dir + '/loss_log.pt')
    psnr_log = torch.load(log_dir + '/psnr_log.pt')
    loss_array_list.append(loss_log[:, -1].numpy()[:max_epoch])
    psnr_array_list.append(psnr_log.numpy()[:max_epoch])

plot_multi_curve(array_list=loss_array_list, label_list=label_list, title='Loss', save='./temp/compare-Loss.pdf')
plot_multi_curve(array_list=psnr_array_list, label_list=label_list, title='PSNR', save='./temp/compare-PSNR.pdf')

# plot_multi_curve(array_list=loss_array_list, label_list=label_list, title='Loss')
# plot_multi_curve(array_list=psnr_array_list, label_list=label_list, title='PSNR')

import torch
import torch.nn.functional as F
import numpy as np
from model.correlation_package import correlation
import time

batch = 8
size = 256
n_feat = 32
radio = 2

corr_func = correlation.Correlation(pad_size=radio, kernel_size=1, max_displacement=radio, stride1=1, stride2=1)

a = np.random.rand(batch, n_feat, size, size)
b = np.random.rand(batch, n_feat, size, size)
a_tensor = torch.FloatTensor(a).cuda()
b_tensor = torch.FloatTensor(b).cuda()

begin = time.time()
cuda_corr = corr_func(a_tensor, b_tensor)
end = time.time()
cuda_time = end - begin

begin = time.time()
a_tensor = F.pad(a_tensor, pad=[radio, radio, radio, radio], mode='constant')
neighbor = []
for i in range(radio * 2 + 1):
    for j in range(radio * 2 + 1):
        neighbor.append(a_tensor[:, :, i:i + size, j:j + size])
a_neighbor = torch.stack(neighbor, dim=1)
b_tensor = b_tensor.view(batch, 1, n_feat, size, size)
torch_corr = a_neighbor * b_tensor
torch_corr = torch.sum(torch_corr, dim=2) / 32.
end = time.time()
torch_time = end - begin
torch.cuda.empty_cache()

diff = torch_corr - cuda_corr
diff_sum = torch.sum(torch.abs(diff))
print('cuda_time', cuda_time)
print('torch_time', torch_time)

print()

# d =2
# corr_func = correlation.Correlation(pad_size=d, kernel_size=1, max_displacement=d, stride1=1, stride2=1)
#
# a = np.array([[[[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]],
#                [[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]],
#                [[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]]
#                ]])
# b = np.array([[[[1, 4, 7],
#                 [2, 5, 8],
#                 [3, 6, 9]],
#                [[1, 4, 7],
#                 [2, 5, 8],
#                 [3, 6, 9]],
#                [[1, 4, 7],
#                 [2, 5, 8],
#                 [3, 6, 9]]
#                ]])
# a_tensor = torch.FloatTensor(a).cuda()
# b_tensor = torch.FloatTensor(b).cuda()
#
# cuda_corr = corr_func(a_tensor, b_tensor)
# cuda_see = cuda_corr[:, :, 1, 1]
#
# a_tensor = F.pad(a_tensor, pad=[d, d, d, d], mode='constant')
# neighbor = []
# for i in range(2*d+1):
#     for j in range(2*d+1):
#         neighbor.append(a_tensor[:, :, i:i + 3, j:j + 3])
# a_neighbor = torch.stack(neighbor, dim=1)
# b_tensor = b_tensor.view(1, 1, 3, 3, 3)
# torch_corr = a_neighbor * b_tensor
# torch_corr = torch.sum(torch_corr, dim=2) / 3.
# torch_see = torch_corr[:, :, 1, 1]
#
# diff = torch_corr - cuda_corr
# diff_sum = torch.sum(torch.abs(diff))
#
# print()

# img0 = './temp/00006.jpg'
# img1 = './temp/00007.jpg'
# img2 = './temp/00008.jpg'
#
# img0_numpy = np.array([imageio.imread(img0)])
# img1_numpy = np.array([imageio.imread(img1)])
# img2_numpy = np.array([imageio.imread(img2)])
#
# img0_tensor = utils.np2Tensor(*img0_numpy, rgb_range=1., n_colors=3)[0]
# c, h, w = img0_tensor.size()
# img0_tensor = img0_tensor.view(1, c, h, w).cuda()
#
# img1_tensor = utils.np2Tensor(*img1_numpy, rgb_range=1., n_colors=3)[0]
# c, h, w = img1_tensor.size()
# img1_tensor = img1_tensor.view(1, c, h, w).cuda()
#
# img2_tensor = utils.np2Tensor(*img2_numpy, rgb_range=1., n_colors=3)[0]
# c, h, w = img2_tensor.size()
# img2_tensor = img2_tensor.view(1, c, h, w).cuda()
#
# net = CorrMap().cuda()
# map = net(img0_tensor, img1_tensor)

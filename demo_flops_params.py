'''

Demo to calculate model's
    Params、Memory、MAdd、Flops、MemR+W

network:
    initial the network

input_size:
    the size of input
    (channel, height, width)

'''

from torchstat import stat

network = None
input_size = (3, 80, 80)

stat(network, input_size)

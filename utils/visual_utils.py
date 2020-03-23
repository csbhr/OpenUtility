import cv2
import numpy as np
import matplotlib.pyplot as plt


def visual_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    # flow shape: [h, w, 2]
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def plot_multi_curve(array_list, label_list, title='compare', xlabel='Epochs', ylabel=None, save=None):
    '''
    function:
        plot multi curve in one figure
    required params:
        array_list: the data of curves, a list
            each item should be numpy.array, ndim=1
        label_list: list, labels for each curve
    optional params:
        title: the title of figure
        xlabel: the flag of x-axis
        ylabel: the flag of y-axis, if None: ylabel=title
        save:
            if None: just show figure
            else: the path to save figure, the figure will be saved
    '''
    assert len(array_list) == len(label_list), "length not equal"

    fig = plt.figure()
    plt.title(title)

    # plot curves
    for i in range(len(array_list)):
        length = array_list[i].shape[0]
        axis = np.linspace(1, length, length)
        plt.plot(axis, array_list[i], label=label_list[i])

    plt.legend()
    plt.xlabel(xlabel)
    if not ylabel:
        ylabel = title
    plt.ylabel(ylabel)
    plt.grid(True)

    if save:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()

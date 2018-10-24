import random
import numpy


def classify(samples):
    '''
    classify samples according to y
    :param samples: samples[x,y], y must be m*1
    :return: the list of classified samples[sample 1,...,sample n]
    '''
    x = samples[0].tolist()     # 把矩阵转换为列表
    y = samples[1].T.tolist()[0]
    y_dict = {}     # 用于记录各类样本在sample_list中的索引
    samples_list = []   # 用于分别存放各类别的样本
    for xi, yi in zip(x, y):    # 遍历原样本集
        if yi not in y_dict.keys():  # 还没有保存过该样本的类别
            ind = len(samples_list)  # 在sample_list添加该类别项
            y_dict[yi] = ind
            samples_list.append([[], []])
        ind = y_dict[yi]    # 将样本放到指定的类别项中
        samples_list[ind][0].append(xi)
        samples_list[ind][1].append(yi)
    for sam in samples_list:    # 将列表转换成矩阵
        sam[0] = numpy.mat(sam[0])
        sam[1] = numpy.mat(sam[1]).T
    return samples_list


def group(samples, k):
    '''
    divide samples into k groups
    :param samples: samples[x,y], y must be m*1
    :param k: the number of groups
    :return: the k_list of groups[group 1,...,group k]
    '''
    temp = []   # 用于暂时存放分组，结构为：每个类别都分为k个组
    samples_list = classify(samples)    # 将原样本集分类
    for sample in samples_list:     # 遍历每个类别
        x = sample[0].tolist()  # 将矩阵转换成列表
        y = sample[1].T.tolist()[0]
        n_g = int(len(x) / k)  # the number of samples each group
        x_group = []    # 用于存放该类别的k各分组
        y_group = []
        for i in range(k):  # 创建好k个分组的空位
            x_group.append([])
            y_group.append([])
        for i in range(len(x)):  # 依次将样本放到特定的空位中
            x_group[i % k].append(x[i])
            y_group[i % k].append(y[i])
        temp.append([x_group, y_group])
    groups = []     # 用于存放分组，结构为：将各类别的k个组统一合并到k个大组
    for i in range(k):  # 创建好k个分组的空位
        groups.append([[], []])
    for te in temp:     # 合并各类的分组
        for i in range(k):
            groups[i][0].extend(te[0][i])
            groups[i][1].extend(te[1][i])
    for i in range(k):  # 将列表转换成矩阵
        groups[i][0] = numpy.mat(groups[i][0])
        groups[i][1] = numpy.mat(groups[i][1]).T
    return groups

import random
import numpy


def classify(samples):
    '''
    classify samples according to y
    :param samples: samples[x,y], y must be m*1
    :return: the list of classified samples[sample 1,...,sample n]
    '''
    x = samples[0].tolist()
    y = samples[1].T.tolist()[0]
    y_dict = {}
    samples_list = []
    for xi, yi in zip(x, y):
        if not yi in y_dict.keys():
            ind = len(samples_list)
            y_dict[yi] = ind
            samples_list.append([[], []])
        ind = y_dict[yi]
        samples_list[ind][0].append(xi)
        samples_list[ind][1].append(yi)
    for sam in samples_list:
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
    temp = []
    samples_list = classify(samples)
    for sample in samples_list:
        x = sample[0].tolist()
        y = sample[1].T.tolist()[0]
        n_g = int(len(x) / k)  # the number of samples each group
        x_group = []
        y_group = []
        for i in range(k):
            x_group.append([])
            y_group.append([])
        for i in range(len(x)):
            x_group[i % k].append(x[i])
            y_group[i % k].append(y[i])
        temp.append([x_group, y_group])
    groups = []
    for i in range(k):
        groups.append([[], []])
    for te in temp:
        for i in range(k):
            groups[i][0].extend(te[0][i])
            groups[i][1].extend(te[1][i])
    for i in range(k):
        groups[i][0] = numpy.mat(groups[i][0])
        groups[i][1] = numpy.mat(groups[i][1]).T
    return groups

import numpy as np
import matplotlib.pyplot as plt


def get_max(arr):
    new_arr = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        new_arr[i] = np.max(arr[:i + 1])
    return new_arr


def get_min(arr):
    new_arr = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        new_arr[i] = np.min(arr[:i + 1])
    return new_arr


def plot_curve(data, title, xlabel='Epochs', ylabel=None, save=None):
    '''
    function:
        plot curve
    required params:
        data: the data of curve, numpy.array, ndim=1
        title: the title for each curve
    optional params:
        xlabel: the flag of x-axis
        ylabel: the flag of y-axis, if None: ylabel=title
        save:
            if None: just show figure
            else: the path to save figure, the figure will be saved
    '''
    length = data.shape[0]
    axis = np.linspace(1, length, length)
    fig = plt.figure()
    plt.title('{} Graph'.format(title))
    plt.plot(axis, data)
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


def plot_multi_curve_given_axis(array_list, label_list, axis, title='compare', xlabel='Epochs', ylabel=None, save=None):
    '''
    function:
        plot multi curve in one figure
    required params:
        array_list: the data of curves, a list
            each item should be numpy.array, ndim=1
        label_list: list, labels for each curve
        axis: list, x-axis flags
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

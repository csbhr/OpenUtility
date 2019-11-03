import numpy as np


def pca_by_k(mat_X, k):
    '''
    PCA降维算法，降到k维
    :param mat_X: 输入矩阵 X（mat m*n）,m是样本数，n是样本的特征数
    :param k: 需要降低的维度（int）
    :return: 特征值（array 4）、特征向量（mat n*k）、矩阵行均值（mat 1*n）
    '''

    # 矩阵中心化
    # 即矩阵每一列都减去该列的均值
    mean_X = mat_X.mean(axis=0)
    mat_X = mat_X - mean_X

    # 计算协方差矩阵
    mat_M = np.dot(mat_X.T, mat_X)

    # 计算特征值、特征矩阵
    # 求出来的特征值、特征矩阵都是按照特征值升序排列
    e, EV = np.linalg.eigh(mat_M)

    # 将特征值、特征矩阵逆序
    e = e[::-1]
    EV = EV[::-1]

    # 返回特征值最大的k个特征值、特征向量、列均值
    return e[:k], EV[:k].T, mean_X


def pca_by_retention(mat_X, retention):
    '''
    PCA降维算法，降到k维
    :param mat_X: 输入矩阵 X（mat m*n）,m是样本数，n是样本的特征数
    :param retention: 需要保留的信息比例，（float 0-1）
    :return: 特征值（array 4）、特征向量（mat n*k）、矩阵行均值（mat 1*n）
    '''

    # 矩阵中心化
    # 即矩阵每一列都减去该列的均值
    mean_X = mat_X.mean(axis=0)
    mat_X = mat_X - mean_X

    # 计算协方差矩阵
    mat_M = np.dot(mat_X.T, mat_X)

    # 计算特征值、特征矩阵
    # 求出来的特征值、特征矩阵都是按照特征值升序排列
    e, EV = np.linalg.eigh(mat_M)

    # 将特征值、特征矩阵逆序
    e = e[::-1]
    EV = EV[::-1]

    # 利用保留信息比例计算需要保留的维度
    sum_e = np.sum(e)
    sum_remain = 0
    k = 0
    for i in range(e.shape[0]):
        sum_remain += e[i]
        k += 1
        if float(sum_remain / sum_e) >= retention:
            break

    # 返回特征值最大的k个特征值、特征向量、列均值
    return e[:k], EV[:k].T, mean_X


def reduce_dimensionality_vector(ori_vector, pca_info):
    '''
    将图片降维
    :param ori_vector: 降维前的向量 (mat 1*n)
    :param pca_info: pca函数得到的特征值、特征向量、列均值 [e, EV, mean_X]
    :return: 降维后的向量（mat 1*k）
    '''
    # mat_pca是PCA降维矩阵(mat n*k)，mean_X(mat 1*n)是矩阵行均值
    mat_pca, mean_X = pca_info[1], pca_info[2]
    # 进行中心化
    ori_vector = ori_vector - mean_X
    # 与PCA降维矩阵相乘，实现降维
    aim_vector = np.dot(ori_vector, mat_pca)
    return aim_vector

import scipy.io as scio
import pandas as pd
import numpy as np


#################################################################################
####                            mat file I/O                                 ####
#################################################################################
def load_mat(file_path):
    return scio.loadmat(file_path)


def save_mat(data, file_path):
    return scio.savemat(file_path, data)


#################################################################################
####                            txt file I/O                                 ####
#################################################################################

def read_txt(file_path):
    '''
    read file into list
    :param file_path: the file's path
    :return: the list of file's content
    '''
    txt_list = []
    with open(file_path) as theFile:
        for line in theFile:
            txt_list.append(line.strip())
    return txt_list


def write_txt(file_path, txt_list):
    '''
    write list into file
    :param file_path: the file's path
    :param txt_list: the list of file's content
    :return: none
    '''
    with open(file_path, "w") as theFile:
        for line in txt_list:
            theFile.write(str(line) + "\n")


def append_txt(file_path, txt_list):
    '''
    append list after file
    :param file_path: the file's path
    :param txt_list: the list of file's content
    :return: none
    '''
    with open(file_path, "a") as theFile:
        for line in txt_list:
            theFile.write(str(line) + "\n")


#################################################################################
####                            csv file I/O                                 ####
#################################################################################
def read_csv(file_path, row_name_ind=None, col_name_ind=None):
    '''
    read csv file into np.array
    :param file_path: the csv file's path
    :param row_name_ind: int, the index of row names
    :param col_name_ind: int, the index of col names
    :return: the np.array of csv file's content
    '''
    csv_data = pd.read_csv(file_path, index_col=row_name_ind, header=col_name_ind)
    row_names = [] if row_name_ind is None else list(csv_data.index)
    col_names = [] if col_name_ind is None else list(csv_data.columns)
    data_array = np.array(csv_data)
    return data_array, col_names, row_names


def write_csv(file_path, data, row_names=None, col_names=None):
    '''
    write list of content into csv file
    :param file_path:  the csv file's path
    :param data: the list of csv file's content
    :param row_names: row names, list, default None
    :param col_names: column names, list, default None
    :return: none
    '''
    content_array = np.array(data)
    index = False if row_names is None else True
    header = False if col_names is None else True
    assert content_array.ndim == 2, 'the ndim of data must be 2!'
    if index:
        assert content_array.shape[0] == len(row_names), 'the number of data rows must equal to len(row_names)'
    if header:
        assert content_array.shape[1] == len(col_names), 'the number of data cols must equal to len(col_names)'
    data = pd.DataFrame(content_array, index=row_names, columns=col_names)
    data.to_csv(file_path, index=index, header=header)

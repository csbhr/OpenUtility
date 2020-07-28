import pandas as pd
import numpy as np


#################################################################################
####                            csv file I/O                                 ####
#################################################################################
def read_csv(file_path, col_name_ind=None, row_name_ind=None):
    '''
    function:
        read csv file into np.array
    required params:
        file_path: the csv file's path
    optional params:
        col_name_ind: int, the index of col names
        row_name_ind: int, the index of row names
    return:
        data_array: the np.array of csv file's content
        col_names: the col names, if col_name_ind=None, return []
        row_name_ind: the row names, if row_name_ind=None, return []
    '''
    csv_data = pd.read_csv(file_path, index_col=row_name_ind, header=col_name_ind)
    row_names = [] if row_name_ind is None else list(csv_data.index)
    col_names = [] if col_name_ind is None else list(csv_data.columns)
    data_array = np.array(csv_data)
    return data_array, col_names, row_names


def write_csv(file_path, data, col_names=None, row_names=None):
    '''
    function:
        write np.array into csv file
    required params:
        file_path: the csv file's path
        data: np.array, the ndim must be 2, the content of csv file
    optional params:
        col_names: a list, default None, if not None, the len must equal to data.shape[1]
        row_names: a list, default None, if not None, the len must equal to data.shape[0]
    '''
    index = False if row_names is None else True
    header = False if col_names is None else True
    assert data.ndim == 2, 'the ndim of data must be 2!'
    if index:
        assert data.shape[0] == len(row_names), 'the number of data rows must equal to len(row_names)'
    if header:
        assert data.shape[1] == len(col_names), 'the number of data cols must equal to len(col_names)'
    data = pd.DataFrame(data, index=row_names, columns=col_names)
    data.to_csv(file_path, index=index, header=header)

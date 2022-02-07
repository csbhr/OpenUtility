import pandas as pd
import numpy as np


#################################################################################
####                            csv file I/O                                 ####
#################################################################################
def read_csv(file_path):
    '''
    function:
        read csv file into list[list[]]
    required params:
        file_path: the csv file's path
    return:
        data_list: the csv file's content
    '''
    csv_data = pd.read_csv(file_path)
    data_list = csv_data.values.tolist()
    return data_list


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


#################################################################################
####                            excel file I/O                                 ####
#################################################################################
def read_excel(file_path, sheet_name=0):
    '''
    function:
        read excel file into list[list[]]
    required params:
        file_path: the excel file's path
    optional params:
        sheet_name: int or str, the index/name of read sheet
    return:
        data_list: the excel file's content
    '''
    excel_data = pd.read_excel(file_path, sheet_name=sheet_name)
    data_list = excel_data.values.tolist()
    return data_list

import os
import pickle
import configparser
import csv


def write_txt(file_path, txt_list):
    '''
    write list into file
    :param file_path: the file's path
    :param txt_list: the list of file's content
    :return: none
    '''
    with open(file_path, "w") as theFile:
        for line in txt_list:
            theFile.write(line + "\n")


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


def get_all_file_by_path(path):
    '''
    get all files' name from the path
    :param path: the path
    :return: the list of all files' name
    '''
    return os.listdir(path)


def pickle_dump(file_path, content_list):
    '''
    dump the list into the file using pickle module
    :param file_path: the file's path
    :param content_list: the list of content
    :return: none
    '''
    with open(file_path, "wb") as f:
        pickle.dump(content_list, f)


def pickle_load(file_path):
    '''
    load the file's content into list using pickle module
    :param file_path: the file's path
    :return: the list of content
    '''
    with open(file_path, "rb") as f:
        content_list = pickle.load(f)
    return content_list


def get_ini_value(file_path, section_name, key_name):
    '''
    get ini file's value
    :param file_path: the file path
    :param section_name: the section name
    :param key_name: the key name
    :return: the value
    '''
    conf = configparser.ConfigParser()
    conf.read(file_path)
    return conf.get(section_name, key_name)

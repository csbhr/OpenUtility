import os
import pickle


#  将txtList中的数据写入到文件fileName中
def writeTxt(filePath, txtList):
    with open(filePath, "w") as theFile:
        for line in txtList:
            theFile.write(line + "\n")


# 将文件fileName中数据读出，以list的形式返回
def readTxt(filePath):
    txtSet = list([])
    with open(filePath) as theFile:
        for line in theFile:
            txtSet.append(line.strip())
    return txtSet


# 获取路径path中所有文件和文件夹名
def get_allFile_byPath(path):
    return os.listdir(path)


# 利用pickle模块，将list对象序列化到文件filePath中
def pickle_dump(filePath, list):
    with open(filePath, "wb") as f:
        pickle.dump(list, f)


# 利用pickle模块，将文件filePath反序列化并返回
def pickle_load(filePath):
    with open(filePath, "rb") as f:
        list = pickle.load(f)
    return list

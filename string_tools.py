def longest_common_subsequence(str1, str2):
    '''
    calculate the length of the longest common subsequence of two strings
    :param str1: string 1
    :param str2: string 2
    :return: the length of the longest common subsequence
    '''
    m, n = len(str1), len(str2)
    ceil = [[0 for i in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            if not i == 0 and not j == 0:
                if str1[i] == str2[j]:
                    ceil[i][j] = ceil[i - 1][j - 1] + 1
                else:
                    ceil[i][j] = max(ceil[i - 1][j], ceil[i][j - 1])
    return ceil[m - 1][n - 1]


def cut_word_indicate(origin_str, indicate_char):
    '''
    cut origin string into words by indicate_char
    :param origin_str: origin string
    :param indicate_char: indicate chars
    :return: the list of word
    '''
    result_list = []

    i = 0   # 去除开头和结尾的非指示字符
    while i < len(origin_str) and origin_str[i] not in indicate_char:
        i = i + 1
    origin_str = origin_str[i:]
    i = len(origin_str) - 1
    while i >= 0 and origin_str[i] not in indicate_char:
        i = i - 1
    origin_str = origin_str[:i + 1]
    if origin_str == "":    # 此字符串中没有指示字符
        return result_list

    str_len = len(origin_str)   # 按非指示字符为分割切割字符串
    i = 0
    start_point = 0
    while i < str_len:
        if origin_str[i] not in indicate_char:
            result_list.append(origin_str[start_point:i])
            j = i
            while j < str_len and origin_str[j] not in indicate_char:
                j = j + 1
            i = j
            start_point = i
        else:
            i = i + 1
    result_list.append(origin_str[start_point:i])
    return result_list

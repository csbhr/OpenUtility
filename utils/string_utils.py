def LCS(str1, str2):
    '''
    Longest Common Subsequence
    '''
    num_res = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
    char_res = [['' for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                num_res[i][j] = num_res[i - 1][j - 1] + 1
                char_res[i][j] = char_res[i - 1][j - 1] + str1[i - 1]
            else:
                if num_res[i][j - 1] > num_res[i - 1][j]:
                    num_res[i][j] = num_res[i][j - 1]
                    char_res[i][j] = char_res[i][j - 1]
                else:
                    num_res[i][j] = num_res[i - 1][j]
                    char_res[i][j] = char_res[i - 1][j]
    return num_res[-1][-1], char_res[-1][-1]

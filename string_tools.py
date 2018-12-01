# 计算两字符串的最长公共子序列长度
def longest_common_subsequence(str1, str2):
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

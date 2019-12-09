def cal_parmeters(model):
    params = list(model.parameters())
    sum = 0
    for i in params:
        layer = 1
        for j in i.size():
            layer *= j
        print("该层的结构：" + str(list(i.size())), "该层参数和：" + str(layer))
        sum = sum + layer
    print("总参数数量和：%.2fM" % (sum / 1e6))
    print("总参数数量和(详细)：", sum)

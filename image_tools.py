import cv2


# 根据百分比获取图片某像素
# position_p是一个列表，position_p[0]是x坐标，position_p[1]是y坐标
def get_pixel_byPercent(img, position_p):
    height, width = img.shape[:2]
    x = int(((height - 1) * position_p[0]) / 100)
    y = int(((width - 1) * position_p[1]) / 100)
    return img[x, y]


# 根据百分比获取图片的像素坐标
# position_p是一个列表，position_p[0]是x坐标，position_p[1]是y坐标
def get_position_byPercent(img, position_p):
    height, width = img.shape[:2]
    x = int(((height - 1) * position_p[0]) / 100)
    y = int(((width - 1) * position_p[1]) / 100)
    return [x, y]


# 将图像缩小为固定大小
# size是缩小后的尺寸，是一个列表，size[0]是高度，size[1]宽度
def resize_image(img, size):
    temp = size[0]
    size[0] = size[1]
    size[1] = temp
    size_tuple = tuple(size)
    return cv2.resize(img, size_tuple, interpolation=cv2.INTER_CUBIC)


# 根据相似度距离排序得到最相似的topNum张图片
def get_top_image_byDistance(list_image, topNum):
    for i in range(0, topNum):
        min = list_image[i].distance
        mink = i
        for j in range(i, len(list_image)):
            if list_image[j].distance < min:
                min = list_image[j].distance
                mink = j
        temp = list_image[i]
        list_image[i] = list_image[mink]
        list_image[mink] = temp
    return list_image[:topNum]

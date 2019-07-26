"""
用mnist数据集制作多目标图片
"""

import os
import cv2
import numpy as np
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images_num", type=int, default=1000)
parser.add_argument("--image_size", type=int, default=416)
parser.add_argument("--images_path", type=str, default="./generate/train")
parser.add_argument("--labels_txt", type=str, default="./yymnist_train.txt")
parser.add_argument("--small", type=int, default=3)  # 一张图片中小目标最多出现几次
parser.add_argument("--medium", type=int, default=6)  # 一张图片中中等目标最多出现几次
parser.add_argument("--big", type=int, default=3)  # 一张图片中大目标最多出现几次
flags = parser.parse_args()

SIZE = flags.image_size  # 生成图片的大小，默认宽和高相等

if os.path.exists(flags.images_path):
    shutil.rmtree(flags.images_path)
os.mkdir(flags.images_path)

# 构建mnist训练集和测试集所有数据的图片地址列表
image_paths = [os.path.join(os.path.realpath("."), "./mnist/train/" + image_name) for image_name in
               os.listdir("./mnist/train")]

image_paths += [os.path.join(os.path.realpath("."), "./mnist/test/" + image_name) for image_name in
                os.listdir("./mnist/test")]


def compute_iou(box1, box2):
    """
    :param box1: (xmin, ymin, xmax, ymax)
    :param box2: (xmin, ymin, xmax, ymax)
    :return:
    """

    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return ((xmax - xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, ratio=1):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]  # 图片的ID
    print(image_path.split("/"))
    image = cv2.imread(image_path)  # 随机从55000z张图片中选取一张
    image = cv2.resize(image, (int(28 * ratio), int(28 * ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, SIZE - w, 1)[0]  # 得到左上角坐标值
        ymin = np.random.randint(0, SIZE - h, 1)[0]
        xmax = xmin + w  # 得到右下角坐标值
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    return blank


with open(flags.labels_txt, "w") as wf:
    # 初始化已制作图片数为0
    image_num = 0

    while image_num < flags.images_num:
        # 获取保存目录的绝对路径，并拼接成为一张图片的绝对路径
        image_path = os.path.realpath(os.path.join(flags.images_path, "%06d.jpg" % (image_num + 1)))

        annotation = image_path
        blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255  # 这样背景为白色，白色RGB值(255,255,255)
        bboxes = [[0, 0, 1, 1]]
        labels = [0]
        data = [blanks, bboxes, labels]
        bboxes_num = 0

        # small object
        ratios = [0.5, 0.8]
        # 随机确定生成图片中有几个小目标
        N = random.randint(0, flags.small)
        if N != 0:
            bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)  # 随机选择一个放缩比例
            idx = random.randint(0, 54999)  # 生成训练数据，生成的随机数包括54999，因此一共有55000种选择
            # idx = random.randint(55000, 64999)  # 生成测试数据，注意三种大小的目标前后保持一致
            data[0] = make_image(data, image_paths[idx], ratio)

        # medium object
        ratios = [1., 1.5, 2.]
        N = random.randint(0, flags.medium)
        if N != 0: bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(0, 54999)
            # idx = random.randint(55000, 64999)
            data[0] = make_image(data, image_paths[idx], ratio)

        # big object
        ratios = [3., 4.]
        N = random.randint(0, flags.big)
        if N != 0:
            bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(0, 54999)
            # idx = random.randint(55000, 64999)
            data[0] = make_image(data, image_paths[idx], ratio)

        if bboxes_num == 0:
            continue

        cv2.imwrite(image_path, data[0])

        for i in range(len(labels)):
            if i == 0:
                continue
            xmin = str(bboxes[i][0])
            ymin = str(bboxes[i][1])
            xmax = str(bboxes[i][2])
            ymax = str(bboxes[i][3])
            class_ind = str(labels[i])
            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
        image_num += 1
        print("=> %s" % annotation)
        wf.write(annotation + "\n")

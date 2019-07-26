"""
显示合成的图片
"""

import cv2
import numpy as np
from PIL import Image

ID = 10  # 选择图片序号
label_txt = "./yymnist_test.txt"
image_info = open(label_txt).readlines()[ID].split()

image_path = image_info[0]
image = cv2.imread(image_path)
print(image.shape)
for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image, (int(float(bbox[0])),
                                  int(float(bbox[1]))),
                          (int(float(bbox[2])),
                           int(float(bbox[3]))), (255, 0, 0), 2)

image = Image.fromarray(np.uint8(image))
image.show()

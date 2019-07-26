## Introduction
A lot of wonderful datasets are now available online, such as COCO or Imagenet. which is challenging the limits of computer vision. But it's not easy for us to do some small experiments with such a large number of images to quickly test the validity of algorithmn. For this reason, I created a small dataset named "yymnist" to do both classification and objection.


| classification | object detection |
|---|---
|![image](./docs/classification.png)|![image](./docs/detection.jpg)|


## Classification

If you want to use this dataset for classification, just use `mnist` folder. Each image in the folder starts with a name of its label. For example:

```
mnist/train/0_32970.pgm -> 0
mnist/train/5_12156.pgm -> 5
```
## Objection Detection

```
$ python yymnist/make_data.py
$ python yymnist/show_image.py # [option]
```
you will defaultly get 1000 pictures in the folder `./generate/train/` and a `yymnist_train.txt` in the root dir. As for `yymnist_train.txt`

```
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
```





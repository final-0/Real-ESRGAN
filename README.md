# Real-ESRGAN

## Abstract
&emsp; You can use this [Real-ESRGAN] to train and test yourself.
This repository is a simple rewrite of the official Real-ESRGAN with as little performance degradation as possible.
The code presented in this repository can perform super-resolution on variety of noises as well as the official Real-ESRGAN.
In other words, super-resolution can be performed not only on low-resolution images compressed by "bicubic", but also on images compressed by "area" and "bilinear".
In addition, super-resolution can be performed on images to which a Gaussian filter is applied.
This function is achieved by learning on low-resolution images with various noises applied.
The generators, discriminators, and loss functions are exactly the same as in the official Real-ESRGAN, allowing for a more sophisticated super-resolution experience.

### default settings <br>
- learning rate : 0.0001
- epochs : 20
- image size : 1024x1024

### training <br>
- esrgan.py <br>

### test <by>
- esrgan_test.py <br>


## Dataset Preparation <br>
dataset : https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset?select=train <br>
&nbsp; recommendation <br>
&emsp; image-size : 1024x1024 <br>
&emsp; number of images : 1000 <br>

## Reference <br>
 github <br>
 &emsp; https://github.com/xinntao/Real-ESRGAN <br>
 paper <br>
 &emsp; https://arxiv.org/abs/2107.10833 <br>

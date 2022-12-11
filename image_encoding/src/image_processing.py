# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)

import skimage 
raw = skimage.io.imread("../images/raw.jpeg")


def show(image,gray=True):
    if gray:
        sns.heatmap(image,cmap='gray',rasterized=True)
#         plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image,rasterized=True)


show(raw,gray=False)
 d

gray = skimage.color.rgb2gray(raw)
show(gray)

gray *= 255
gray = gray.astype(int)
show(gray)

gray.shape

# # 无损编码/压缩算法

# ## 行程编码压缩

pre = gray[0][0]
l = 0
run_length = []
for pixel in gray.flatten():
    if pixel == pre:
        l += 1
    else:
        run_length.append([pre,l])
        l = 1
        pre = pixel

run_length = np.array(run_length)
run_length.shape

ratio = len(run_length.flatten()) / len(gray.flatten())
ratio

# ## 实现哈夫曼压缩

# # 有损压缩/压缩算法实验
# -  [影像算法解析——JPEG 压缩算法](https://zhuanlan.zhihu.com/p/40356456)

show(raw,gray=False)

# ## RGB 到 YCBCR 的转换 

ycbcr = skimage.color.rgb2ycbcr(raw)
ycbcr

# ## analyse



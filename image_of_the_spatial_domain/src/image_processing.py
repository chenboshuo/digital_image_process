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

import skimage 
raw = skimage.io.imread("../images/raw.jpg")


def show(image,gray=True):
    if gray:
        sns.heatmap(image,cmap='gray',rasterized=True)
#         plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image,rasterized=True)


show(raw,gray=False)

# # 计算图像有关的统计参数

# https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html
gray = skimage.color.rgb2gray(raw)
show(gray)

gray.shape

import pandas as pd

df = pd.Series(gray.flatten())
df.describe()

df.plot(kind='kde')
plt.xlim(0,1)
plt.savefig("../figures/kde.pdf")

# # 处理

# ## 添加高斯噪声

import numpy as np

mean = 0
sigma = 0.1
gaussian = np.random.normal(mean, sigma, (raw.shape[0],raw.shape[1])) 
noisy = gray + gaussian 

noisy[noisy<0] = 0
noisy[noisy>255] = 255

show(noisy)

# +
plt.subplot(2,1,1)
show(gray)
plt.subplot(2,1,2)
show(noisy)

plt.savefig("../figures/noisy.pdf",bbox_inches="tight")
# -

df = pd.DataFrame({"raw": gray.flatten() , "noisy": noisy.flatten()})
df.describe()

noisy_flattern = noisy.copy()
noisy_flattern.shape = noisy.shape[0]*noisy.shape[1]

df.plot(kind='kde')
plt.savefig("../figures/noisy_raw.pdf")

# ## 改变图像大小

# ### 改变矩阵大小

raw.shape

half = raw.copy()
half.shape = raw.shape[0]*2, raw.shape[1]//2,3
show(half,gray=False)

# ### 使用 skimage.transform.rescale

# https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html
shrink = skimage.transform.rescale(gray,0.25) 
shrink.shape

show(shrink)
plt.savefig("../figures/shrink.pdf")

df = pd.DataFrame.from_dict({"raw":gray.flatten(), "rotated":shrink.flatten()},orient="index").T

df.describe()

df.plot(kind='kde')

# ## 旋转图像

rotated = skimage.transform.rotate(gray,30)
show(rotated)
plt.savefig("../figures/rotated.pdf")

pd.DataFrame({"raw":gray.flatten(), "rotated":rotated.flatten()}).plot(kind='kde')
plt.savefig("../figures/rotated_kde.pdf")

# ## 裁剪

clipped = gray[:gray.shape[0]//2,:]


# +
plt.subplot(2,1,1)
show(gray)
plt.subplot(2,1,2)
plt.ylim(0,1080)
show(clipped)

plt.savefig("../figures/clipped.pdf",bbox_inches="tight")
# -

df = pd.DataFrame.from_dict({"raw":gray.flatten(), "clipped":clipped.flatten()},orient="index").T
df.describe()

df.plot(kind='kde')
plt.savefig("../figures/clipped_kde.pdf")

# # 增强操作

# ## 图像的对比度变换

# ### RGB

# +
from ipywidgets import interact

def update(gamma = 1.0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    gamma_corrected = skimage.exposure.adjust_gamma(raw, gamma)
    ax.set_title(f"$\gamma$ = {gamma}")
    ax.imshow(gamma_corrected)

    fig.canvas.draw()

interact(update,gamma=(0.1,3));
# -

gamma_corrected_light = skimage.exposure.adjust_gamma(raw, 0.5)
show(gamma_corrected_light,gray=False)

gamma_corrected_dark = skimage.exposure.adjust_gamma(raw, 3)
show(gamma_corrected_dark,gray=False)

# +
plt.subplot(3,1,1)
show(gamma_corrected_light, gray=False)

plt.subplot(3,1,2)
show(raw, gray=False)

plt.subplot(3,1,3)
show(gamma_corrected_dark, gray=False)

plt.savefig("../figures/gamma.pdf",bbox_inches="tight")
# -
# ### Gray


# +

def update(gamma = 1.0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    gamma_corrected = skimage.exposure.adjust_gamma(gray, gamma)
    ax.set_title(f"$\gamma$ = {gamma}")
    ax.imshow(gamma_corrected,cmap="gray")

    fig.canvas.draw()

interact(update,gamma=(0.1,3));
# -

gamma_corrected_light = skimage.exposure.adjust_gamma(gray, 0.5)
show(gamma_corrected_light)

gamma_corrected_dark = skimage.exposure.adjust_gamma(gray, 3)
show(gamma_corrected_dark)

# +
plt.subplot(3,1,1)
show(gamma_corrected_light)

plt.subplot(3,1,2)
show(gray)

plt.subplot(3,1,3)
show(gamma_corrected_dark)

plt.savefig("../figures/gamma_gray.pdf",bbox_inches="tight")
# -

# ## 均值滤波

size = 50

# https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_structuring_elements.html#sphx-glr-auto-examples-numpy-operations-plot-structuring-elements-py
kernel = skimage.morphology.square(size)
kernel

# https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.mean
avg = skimage.filters.rank.mean(gray, footprint=kernel)
show(avg)

# +
plt.subplot(3,1,1)
show(gray)

plt.subplot(3,1,2)
show(skimage.filters.rank.mean(gray, footprint= skimage.morphology.square(20)))

plt.subplot(3,1,3)
show(avg)
plt.savefig("../figures/avg.pdf",bbox_inches="tight")


# +
def update(size = 5):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    kernel = skimage.morphology.square(size)
    avg = skimage.filters.rank.mean(gray, footprint=kernel)
    ax.set_title(f"size = {size}")
    ax.imshow(avg,cmap="gray")

    fig.canvas.draw()

interact(update,size=(0,100));
# -
# ## 锐化


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)

import scipy.ndimage as ndi

# bing 2022-10-30
fish = skimage.io.imread("../images/fish.jpeg")
# fish = skimage.io.imread("../images/Lenna.png")
show(fish, gray=False)
plt.savefig("../figures/fish.pdf")

fish.shape

gray = skimage.color.rgb2gray(fish)
show(gray)

# ### Sobel

kernel_x = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]])
x_sharpen = ndi.convolve(gray, kernel_x)
show(x_sharpen)

kernel_y = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]])
y_sharpen = ndi.convolve(gray,kernel_y)
show(y_sharpen)

show((x_sharpen ** 2 + y_sharpen ** 2) ** 1/2)

sobel_abs = abs(x_sharpen) + abs(y_sharpen)
show(sobel_abs)

sobel = skimage.filters.sobel(gray)
show(sobel)

pd.DataFrame({"sobel": sobel_abs.flatten(), "skimage.filter.sobel": sobel.flatten()}).plot(kind="kde")

# +
plt.subplot(2,2,1)
show(x_sharpen)

plt.subplot(2,2,2)
show(y_sharpen)

plt.subplot(2,2,3)
show(sobel_abs)

plt.subplot(2,2,4)
show(sobel)

plt.savefig("../figures/sobel.pdf",bbox_inches="tight")
# -

# ### personal kernel

# $$
#     G'x = \begin{bmatrix}
#         -1 & 0 & 1 \\
#         -1 & 0 & 1 \\
#         -1 & 0 & 1
#     \end{bmatrix}
#     \circledast A 
# $$
#
# $$
#     G'y = \begin{bmatrix}
#         1 & 1 & 1 \\
#         0 & 0 & 0 \\
#         -1 & -1 & -1
#     \end{bmatrix}
#     \circledast A 
# $$
#
# $$
#     A'(x,y) = |G'_x(x,y)| + |G'_y(x,y)| 
# $$

kernel_x = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]])
x_sharpen = ndi.convolve(gray, kernel_x)
show(x_sharpen)

kernel_y = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]])
y_sharpen = ndi.convolve(gray,kernel_y)
show(y_sharpen)

new_abs = abs(x_sharpen) + abs(y_sharpen)
show(new_abs)

# +
plt.subplot(2,2,1)
show(sobel_abs)

plt.subplot(2,2,2)
show(new_abs)

plt.savefig("../figures/sobel_vs_personal.pdf",bbox_inches="tight")
# -

pd.DataFrame({"sobel": sobel_abs.flatten(), "personal": new_abs.flatten()}).plot(kind="kde")
plt.savefig("../figures/sobel_compare.pdf")

# ### design a kernel

kernel_13 = np.array([
    [0,0,1],
    [0,0,0],
    [-1,0,0]])
sharpen_13 = abs(ndi.convolve(gray, kernel_13))
show(sharpen_13)

kernel_24 = np.array([
    [1,0,0],
    [0,0,0],
    [0,0,-1]])
sharpen_24 = abs(ndi.convolve(gray, kernel_24))
show(sharpen_24)

# +
plt.subplot(2,2,1)
show(sharpen_13)

plt.subplot(2,2,2)
show(sharpen_24)

plt.savefig("../figures/sharpen_13_24.pdf",bbox_inches="tight")
# -

show(sharpen_13+sharpen_24)
plt.savefig("../figures/sharpen_13+24.pdf",bbox_inches="tight")



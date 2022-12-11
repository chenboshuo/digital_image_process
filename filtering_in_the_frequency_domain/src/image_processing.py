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
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)

import skimage 
raw = skimage.io.imread("../images/raw.jpg")


def show(image,gray=True):
    if gray:
        sns.heatmap(image,cmap='gray',rasterized=True)
#         plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image,rasterized=True)


def show_f(frequency_val,vmin=-500,vmax=500,cmap = 'coolwarm'):
    sns.heatmap(frequency_val.astype(float),cmap=cmap,
                rasterized=True,
                vmin=vmin, vmax=vmax)


show(raw,gray=False)
plt.savefig("../figures/raw.pdf")

gray = skimage.color.rgb2gray(raw)
show(gray)
plt.savefig("../figures/gray.pdf")

# # 灰度图像的频域滤波器

# ## 灰度图像进行离散傅里叶变换

freq = np.fft.fft2(gray)
freq

shifted = np.fft.fftshift(freq)

d = pd.DataFrame(shifted.flatten())
d.plot(kind='hist')

d.plot(kind='hist')
plt.xlim(-100000,100000)

real = shifted.copy().astype(float)
real[real > 500] = 500
real[real < -500] = -500
show_f(real)
plt.savefig("../figures/freq.pdf")

# ## 梯形低通滤波
#
# $$
# H(u,v) = \begin{cases}
# 1 & D \in (-\infty,D_0) \\
# \frac{D(u,v)-D_1}{D_0 - D_1} & D \in (D_0, D_1) \\
# 0 & D \in (D_1,+\infty)
# \end{cases}
# $$
#
# $$
# D = \sqrt{u^2 + v^2}
# $$

# $$
# \begin{cases}
#   D_0 = 10 \\
#   D_1 = 80
# \end{cases}
# $$

# ### raw mask

line_size, col_size = shifted.shape
u = abs(np.arange(-col_size//2, col_size//2))
v = abs(np.arange(-line_size//2, line_size//2))
u,v = np.meshgrid(u,v)
raw_mask = (u*u + v*v)**(1/2)
show_f(raw_mask,vmax = 1500, vmin = 0,cmap='Blues')
plt.savefig("../figures/raw_mask.pdf")

# 引入
# $$
# D' = \frac{D-D_1}{D_0 - D_1}
# $$
# 则
# $$
# H = \begin{cases}
#   1 & D' \in (1,+\infty) \\
#   D' & D' \in (0,1) \\
#   0 & D' \in (-\infty,0)
# \end{cases}
# $$

# ### case 1

low = 10
high = 80

mask = raw_mask.copy()
mask -= high
mask /= (low-high)
mask[mask > 1] = 1
mask[mask < 0] = 0
show_f(mask,vmin=0,vmax=1,cmap='Blues')
plt.savefig("../figures/low_pass_mask.pdf")

low_pass_freq = shifted * mask
show_f(low_pass_freq)
plt.savefig("../figures/low_pass_freq.pdf")

filtered = np.fft.ifftshift(low_pass_freq)
filtered = np.fft.ifft2(filtered)
filtered = abs(filtered)
show(filtered)
plt.savefig("../figures/low_pass_filter.pdf")

plt.subplot(1,2,1)
show(gray[:100,:30])
plt.subplot(1,2,2)
show(filtered[:100,:30])
plt.savefig("../figures/ringingeffect.pdf")

# ### case 2

low = 100
high = 300

mask = raw_mask.copy()
mask -= high
mask /= (low-high)
mask[mask > 1] = 1
mask[mask < 0] = 0
show_f(mask,vmin=0,vmax=1,cmap='Blues')
plt.savefig("../figures/low_pass_mask_2.pdf")

low_pass_freq = shifted * mask
show_f(low_pass_freq)
plt.savefig("../figures/low_pass_freq_2.pdf")

filtered = np.fft.ifftshift(low_pass_freq)
filtered = np.fft.ifft2(filtered)
filtered = abs(filtered)
show(filtered)
plt.savefig("../figures/low_pass_filter_2.pdf")

plt.subplot(1,2,1)
show(gray[:100,:30])
plt.subplot(1,2,2)
show(filtered[:100,:30])
plt.savefig("../figures/ringingeffect_2.pdf")

# ## 梯形高通滤波

# $$
# H(u,v) = \begin{cases}
# 0 & D \in (-\infty,D_1) \\
# \frac{D(u,v)-D_1}{D_0 - D_1} & D \in (D_1, D_0) \\
# 1 & D \in (D_0,+\infty)
# \end{cases}
# $$
#
# 引入
# $$
# D' = \frac{D-D_1}{D_0 - D_1}
# $$
# 则
# $$
# H = \begin{cases}
#   0 & D' \in (-\infty,0) \\
#   D' & D' \in (0,1) \\
#   1 & D' \in (0,+\infty)
# \end{cases}
# $$

# ### case 1

low = 10
high = 80

mask = raw_mask.copy()
mask -= low
mask /= (high-low)
mask[mask > 1] = 1
mask[mask < 0] = 0
show_f(mask,vmin=0,vmax=1,cmap='Blues')
plt.savefig("../figures/high_pass_mask_1.pdf")

low_pass_freq = shifted * mask
show_f(low_pass_freq)
plt.savefig("../figures/high_pass_freq_1.pdf")

filtered = np.fft.ifftshift(low_pass_freq)
filtered = np.fft.ifft2(filtered)
filtered = abs(filtered)
show(filtered)
plt.savefig("../figures/high_pass_filter_1.pdf")

# ### case 2

low = 100
high = 300

mask = raw_mask.copy()
mask -= low
mask /= (high-low)
mask[mask > 1] = 1
mask[mask < 0] = 0
show_f(mask,vmin=0,vmax=1,cmap='Blues')
plt.savefig("../figures/high_pass_mask_2.pdf")

low_pass_freq = shifted * mask
show_f(low_pass_freq)
plt.savefig("../figures/high_pass_freq_2.pdf")

low_pass_freq = shifted * mask
show_f(low_pass_freq)
plt.savefig("../figures/high_pass_freq_2.pdf")

filtered = np.fft.ifftshift(low_pass_freq)
filtered = np.fft.ifft2(filtered)
filtered = abs(filtered)
show(filtered)
plt.savefig("../figures/high_pass_filter_2.pdf")

# # 灰度图像的离散余弦变换

# ## case 1

cell = 8
freq = gray.copy()
outcome = gray.copy()

# -   [scipy.fft.dctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html)
# -   [scipy.fft.dct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)

import itertools
for i,j in itertools.product(
        range(cell, line_size, cell), 
        range(cell, col_size,cell)):
    freq[i-cell:i, j-cell:j] = scipy.fft.dctn(gray[i-cell:i, j-cell:j])
    outcome[i-cell:i, j-cell:j] = scipy.fft.idctn(freq[i-cell:i, j-cell:j])

pd.DataFrame({'DCT II':freq.flatten()}).plot(kind='hist')

show_f(freq,vmax=50,vmin=-50)
plt.savefig("../figures/dct_II_freq.pdf")

show(outcome)

dct_stat = pd.DataFrame({'diff':gray.flatten()-outcome.flatten()})
dct_stat.describe()

dct_stat.plot(kind='hist')
plt.savefig("../figures/dct_III_diff_hist.pdf")

show_f(outcome - gray,vmin = -5e-16, vmax=5e-16)
plt.savefig("../figures/dct_II_diff.pdf")

# ## case 2

for i,j in itertools.product(
        range(cell, line_size, cell), 
        range(cell, col_size,cell)):
    freq[i-cell:i, j-cell:j] = scipy.fft.dctn(
        gray[i-cell:i, j-cell:j],
        type=3
    )
    outcome[i-cell:i, j-cell:j] = scipy.fft.idctn(freq[i-cell:i, j-cell:j],type=3)

pd.DataFrame({'DCT III':freq.flatten()}).plot(kind='hist')

show_f(freq,vmax=50,vmin=-50)
plt.savefig("../figures/dct_III_freq.pdf")

show(outcome)

dct_stat = pd.DataFrame(gray.flatten()-outcome.flatten())
dct_stat.describe()

dct_stat.plot(kind='hist')
plt.savefig("../figures/dct_III_hist.pdf")

show_f(outcome - gray,vmin = -1e-15, vmax=1e-15)
plt.savefig("../figures/dct_III_diff.pdf")





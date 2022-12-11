import cv2 as cv
import matplotlib.pyplot as plt

raw = cv.imread("../images/raw.jpeg")

plt.imshow(raw,rasterized=True)



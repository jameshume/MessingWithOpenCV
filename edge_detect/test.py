import cv2
import numpy as np
from matplotlib import pyplot as plt

orignalImg = cv2.cvtColor(cv2.imread('hyundai-verna.jpg',), cv2.COLOR_BGR2GRAY)
gaussImg = cv2.GaussianBlur(orignalImg, (5, 5), 0)

fig, axs = plt.subplots(nrows = 3, ncols=2)
imgList = [(axs[0][0], "Original", orignalImg),
           (axs[0][1], "Blurred", gaussImg),
           (axs[1][0], "Lapacian", cv2.Laplacian(gaussImg, cv2.CV_64F)),
           (axs[1][1], "Sobel X", cv2.Sobel(gaussImg, cv2.CV_64F, dx=1, dy=0, ksize=5)),
           (axs[2][0], "Sobel Y", cv2.Sobel(gaussImg, cv2.CV_64F, dx=0, dy=1, ksize=5)),
           (axs[2][1], "Canny", cv2.Canny(gaussImg, 10, 70))]
for ax, descr, img in imgList:
    ax.imshow(img, cmap = 'gray')
    ax.axis('off')
    ax.set_title(descr)

fig.show()
plt.show()
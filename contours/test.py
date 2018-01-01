import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('hyundai-verna.jpg')
imBlank = np.ones(im.shape)
imGray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imThresh, contours, hierarchy = \
    cv2.findContours(
        cv2.threshold(imGray, 100, 255, 0)[1], 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imBlank, contours, -1, (0,0,0), 1)

fig, axs = plt.subplots(nrows = 2, ncols=2)
axs[0][0].imshow(im, cmap = 'gray'),       axs[0][0].axis('off')
axs[0][1].imshow(imGray, cmap = 'gray'),   axs[0][1].axis('off')
axs[1][0].imshow(imThresh, cmap = 'gray'), axs[1][0].axis('off')
axs[1][1].imshow(imBlank, cmap = 'gray'),  axs[1][1].axis('off')

fig.show()
plt.show()
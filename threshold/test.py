import cv2 
import matplotlib.pyplot as pl

myImage = cv2.imread('rare-coins.jpg', cv2.IMREAD_GRAYSCALE)

blackPixels = myImage < 255
whitePixels = ~blackPixels

fig, axs = pl.subplots(ncols=2)
axs[0].axis('off')
axs[0].imshow(cv2.cvtColor(myImage, cv2.COLOR_GRAY2RGB))
myImage[blackPixels] = 0
myImage[whitePixels] = 255
axs[1].axis('off')
axs[1].imshow(cv2.cvtColor(myImage, cv2.COLOR_GRAY2RGB))

fig.show()
pl.show()



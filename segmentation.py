import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Optical Images/H13/A1/bottom_1.tif")
img_resized = cv2.resize(img,(1024,768))
# cv2.imshow('img',img_resized)
# cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
# thresh = thresh.astype(np.uint32)


# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True) 

# Select long perimeters only
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
# listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
# numcards=len(listindex)

card_number = -1 #just so happened that this is the worst case
stencil = np.zeros(img.shape).astype(img.dtype)
cv2.drawContours(stencil, [contours[listindex[card_number]]], 0, (255, 255, 255), cv2.FILLED)
res = cv2.bitwise_and(img, stencil)
cv2.imwrite("out.bmp", res)
canny = cv2.Canny(res, 100, 200)
cv2.imwrite("canny.bmp", canny)
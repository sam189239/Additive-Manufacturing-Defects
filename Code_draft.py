import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the original image
# img = cv2.imread(r"C:\Users\Abhishek\Desktop\test\bottom_1.tif")
img = cv2.imread(r"Optical Images - JAM Lab\H13\A1\bottom_1.tif")

# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0) 


img_resized = cv2.resize(img,(1024,768))

# cv2.imshow('img',img_resized)
# cv2.waitKey(0)

# Convert to graycsale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img = cv2.GaussianBlur(img, (3,3), 0)

#incresing contrast before thresholding
# clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
# l, a, b = cv2.split(lab)  # split on 3 different channels

# l2 = clahe.apply(l)  # apply CLAHE to the L-channel

# lab = cv2.merge((l2,a,b))  # merge channels
# img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
# cv2.imshow('Increased contrast', img2)
# #cv2.imwrite('sunset_modified.jpg', img2)

# cv2.waitKey(0)

## Simple Thresholding
ret,thresh1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,50,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,50,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,50,255,cv2.THRESH_TOZERO_INV)
titles = ['Given Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
plt.figure()
for i in range(6):
   plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
 
# Sobel Edge Detection
#sobelx = cv2.Sobel(src=thresh1, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
#sobely = cv2.Sobel(src=thresh1, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
#sobelxy = cv2.Sobel(src=thresh1, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images
# cv2.imshow('Sobel X', sobelx)
# cv2.waitKey(0)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=thresh1, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)
# cv2.waitKey(0)
 
# cv2.destroyAllWindows()

# Image dilation
img_dilation = cv2.dilate(edges, None, iterations=1)
# cv2.imshow('Dilation', img_dilation)
 
# cv2.waitKey(0)

# Finding total Contour in image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("number of contours = " + str(len(contours)))
largest_contour = 0

contour_img = np.zeros(np.shape(img))

for i in range(len(contours)):
   cv2.drawContours(contour_img,contours,i,(255,0,0),3)
   if np.shape(contours[i])[0]>largest_contour:
      largest_contour = np.shape(contours[i])[0]
      largest_contr_idx = i

# # Individual Contours in image
# for j in range(contours):
#    cv2.drawContours(img,contours,largest_contr_idx,(255,0,0),3)

# Find circles
import numpy as np
# output = img.copy()
# output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
op = np.zeros(np.shape(img))
circles = cv2.HoughCircles(img_dilation, cv2.HOUGH_GRADIENT, 1.3, 100)
# If some circle is found
if circles is not None:
   # Get the (x, y, r) as integers
   circles = np.round(circles[0, :]).astype("int")
   print(circles)
   # loop over the circles
   for (x, y, r) in circles:
      cv2.circle(output, (x, y), r, (0, 255, 0), 2)
# show the output image


contour_list = []
count = 0
for contour in contours:
   approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
   area = cv2.contourArea(contour)
   if ((len(approx) > 12) & (area > 0)): #& cv2.isContourConvex(approx)
      count+=1
      contour_list.append(contour)


for contour in contours:
   # median = 
   approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
   area = cv2.contourArea(contour)
   if ((len(approx) > 12) & (area > 0)): #& cv2.isContourConvex(approx)
      print(np.shape(contour))


print(count)
cv2.drawContours(op, contour_list,  -1, (255,0,0), 2)


plt.figure()
plt.subplot(221)
plt.imshow(edges, cmap='gray')
plt.title('Edge')
plt.subplot(222)
plt.imshow(img_dilation, cmap='gray')
plt.title('Dilated')
plt.subplot(223)
plt.imshow(contour_img, cmap='gray')
plt.title('Contours')
plt.subplot(224)
plt.imshow(op, cmap='gray')
plt.title('Circles')
plt.show()

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
   
#     if len(approx)==8:
#         area = cv2.contourArea(cnt)
#         (cx, cy), radius = cv2.minEnclosingCircle(cnt)
#         circleArea = radius * radius * np.pi
#         print(circleArea)
#         print(area)
#         if circleArea == area:
#             cv2.drawContours(img, [cnt], 0, (220, 152, 91), -1)


# TODO #

#update doc 
#hough didnt work, number of edges of 12, iscontourconvex not working properly
#put output in doc
#median of all points, mean of distance and devioation , deviation divided by mean, set up threshold, set up a threshold
#area print area threshold
#physics behind contour convex, findcontours
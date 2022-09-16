import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread("Optical Images/H13/A1/bottom_1.tif")
img_resized = cv2.resize(img,(1024,768))
# cv2.imshow('img',img_resized)
# cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# a = np.asarray(img)

# dst = np.zeros(shape=(5,2))

# b=cv2.normalize(a,dst,0,255,cv2.NORM_L1)


# im = Image.fromarray(b)

norm_image = cv2.resize(norm_image, (1024,768))
norm_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("normalized image", norm_image)
cv2.imshow("image", img_resized)

cv2.waitKey(0)
# titles = ['Original Image', 'Normalized Image']
# images = [img, norm_image]
# for i in range(2):
#     plt.subplot(1,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
cv2.destroyAllWindows()
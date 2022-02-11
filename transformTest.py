# import cv2
# from operator import itemgetter
# from glob import glob
# import matplotlib.pyplot as plt
# import numpy as np

# image = cv2.imread("perspectiveBook.jpg")

# width = int(image.shape[1] * 40 / 100)
# height = int(image.shape[0] * 40 / 100)
# dim = (width, height)

# paper = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# cv2.imshow("test", paper)
# #---- Original Coordinates for object ----#
# pts1 = np.float32([[158,203],[352,92],[480,396],[655,223]])
# #---- Destination Coordinates for object later ----#
# pts2 = np.float32([[300,0],[710,0],[300,526],[710,526]])
# # for val in pts1:
# #     cv2.circle(paper,(val[0],val[1]),5,(0,255,0),-1)

# #---- Getting the transformation matrix ----#
# M = cv2.getPerspectiveTransform(pts1,pts2)
# #---- Applies perspective transformation to an image ----#
# dst = cv2.warpPerspective(paper,M,(850,530))

# cv2.imshow("result",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


import cv2
from operator import itemgetter
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("rcam.jpg")

width = int(image.shape[1] * 60 / 100)
height = int(image.shape[0] * 60 / 100)
dim = (width, height)

paper = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


cv2.imshow("test", paper)
#---- Original Coordinates for object ----#
pts1 = np.float32([[152,201],[457,201],[11,300],[508,300]])
#---- Destination Coordinates for object later ----#
pts2 = np.float32([[0,0],[710,0],[0,526],[610,526]])
# for val in pts1: 
#     cv2.circle(paper,(val[0],val[1]),5,(0,255,0),-1)

#---- Getting the transformation matrix ----#
M = cv2.getPerspectiveTransform(pts1,pts2)
print("M VALUES : \n ",M)
#---- Applies perspective transformation to an image ----#
dst = cv2.warpPerspective(paper,M,(650,530))

cv2.imshow("result",dst)
cv2.waitKey()
cv2.destroyAllWindows()
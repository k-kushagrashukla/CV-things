# for resizing the large image
import cv2

img=cv2.imread("vk2.jpg")
rsize=cv2.resize(img,(200,200),interpolation=cv2.INTER_AREA)
cv2.imshow("original image",rsize)
cv2.waitKey(0)
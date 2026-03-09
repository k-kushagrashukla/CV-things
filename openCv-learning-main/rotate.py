# for rotating an image
import cv2

img=cv2.imread("vk2.jpg")

width=img.shape[0]
height=img.shape[1]
centre=(height/2,width/2)
# 30 is nothing but how much degree is given to move a photo
m=cv2.getRotationMatrix2D(centre,30,1)
img1=cv2.warpAffine(img,m,(height,width))

cv2.imshow("original image",img)
cv2.imshow("rotated image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for blurring an image in cv
import cv2

img=cv2.imread("vk2.jpg")

gblur=cv2.GaussianBlur(img,(111,111),0)

cv2.imshow("gaussian blur",gblur)
cv2.imshow("original image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
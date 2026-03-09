# making an border in img 
import cv2

img=cv2.imread("vk2.jpg")

border=cv2.copyMakeBorder(img,20,20,20,20,borderType=cv2.BORDER_CONSTANT)

cv2.imshow("border image",border)

cv2.waitKey(0)
cv2.destroyAllWindows()
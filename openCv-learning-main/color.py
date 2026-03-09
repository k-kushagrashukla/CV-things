import cv2

img=cv2.imread("vk2.jpg")

img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imshow("originali image",img)
cv2.imshow("new image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
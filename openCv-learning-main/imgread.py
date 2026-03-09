import cv2
img=cv2.imread("vk2.jpg")
# for saving it again to the editor
cv2.imwrite("savedimage.png",img)
cv2.imshow("image",img)
cv2.waitKey(0)


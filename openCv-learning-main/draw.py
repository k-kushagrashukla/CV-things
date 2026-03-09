import cv2

img=cv2.imread("vk2.jpg")

#line=cv2.line(img,(0,0),(500,700),(255),10)

#for writing text
#text=cv2.putText(img,"niharika",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,255,2)
#rectangle=cv2.rectangle(img,(0,0),(500,700),255,10)
circle=cv2.circle(img,(250,350),150,255,thickness=10)
cv2.imshow("image",circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
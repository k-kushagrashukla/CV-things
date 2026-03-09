import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('face.xml')

webcam=cv2.VideoCapture(0)
success,img=webcam.read()



graying=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


facecordinates=trainedData.detectMultiScale(graying)

for i in range(1):
    x,y,w,h=facecordinates[i]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#for x,y,w,h in facecordinates:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256),2))

cv2.imshow('window',img)
cv2.waitKey()

print('end of program')
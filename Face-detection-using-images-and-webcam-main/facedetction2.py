import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('face.xml')
img=cv2.imread('hero.jpg')
graying=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

facecordinates=trainedData.detectMultiScale(graying)

#for i in range(0,8):
    #x,y,w,h=facecordinates[i]
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

for x,y,w,h in facecordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256),5))


cv2.imshow('window',img)
cv2.waitKey()

print('end of program')
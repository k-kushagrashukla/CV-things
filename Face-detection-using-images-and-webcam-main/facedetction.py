import cv2
trainedData= cv2.CascadeClassifier('face.xml')

img=cv2.imread('shr.jpg')

#coversion to greyscale
graying= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
facecordinates=trainedData.detectMultiScale(graying)

x,y,w,h=facecordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('window',img)
cv2.waitKey()

print("end of program")
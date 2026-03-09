import cv2
face_detect=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_detect=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

img=cv2.imread("vk2.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face=face_detect.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in face:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),255,2)
    eye_gray=gray[y:y+h,x:x+w]
    eye_color=img[y:y+h,x:x+w]
    eye=eye_detect.detectMultiScale(eye_gray)
    for (x1,y1,h1,w1) in eye:
        cv2.rectangle(eye_color,(x1,y1),(x1+w1,y1+h1),255,2)

cv2.imshow("detected face and eye",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
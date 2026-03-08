import numpy as np
import cv2
# chechre pehchane ke liye pretend model
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#game get started or not
first_read=True

cap=cv2.VideoCapture(0)
# camere se frame tasveer ko padhta hai . ret boolean btataa hai ki frame sahi se padha gya ki nhi. last img mein frame store hota hai
ret,img=cap.read()
# jab tak camera frame dega tab tak return true hai 
while(ret):
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # bilateral image ko smooth karta hai jisse noises hat jaaye but edges bachi rahe
    gray=cv2.bilateralFilter(gray,5,1,1)
    # detectmulti scale image mei chehre ka pta lagata hai 
    faces=face_cascade.detectMultiScale(gray,1.3,5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            # drawing rectangle . x,y face ka shuruaati hissa . (x+w,y+h) faace ka end ka hissa.
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # face ki jo grayscale mein value hai vahi same normal image mein define kiya gya hai.
            roi_face=gray[y:y+h,x:x+w]
            roi_face_clr=img[y:y+h,x:x+w]
            eyes= eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
            if(len(eyes)>=2):
                if(first_read):
                    # text ki position(100,100)
                    cv2.putText(img,"press s to begin",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                else:
                    print("-------------")
            else:
                if(first_read):
                    cv2.putText(img,"no eyes detected",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
                else:
                    print("you loose")
                    first_read=True
    else:
        cv2.putText(img,"no face detected",(100,100),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    cv2.imshow('img',img)
    # ek keyborad input ka wait karega tabhi window close hogi.
    a=cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        first_read=False

#web cam ko release band kart hai.
cap.release()
cv2.destroyAllWindows()
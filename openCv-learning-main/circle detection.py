import cv2
import numpy as np
img=cv2.imread("ec.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_blurred=cv2.blur(gray,(3,3))
detectcircle=cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT,1,100)

if detectcircle is not None:
    detectcircle=np.uint16(np.around(detectcircle))
    for points in detectcircle[0,:1]:
        a,b,r=points[0],points[1],points[2]
        cv2.circle(img,(a,b),r,255,2)
        cv2.imshow("detected circle", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
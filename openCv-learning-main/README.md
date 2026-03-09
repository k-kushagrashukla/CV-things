# OpenCV
OpenCV ka full form hai Open Source Computer Vision. Yeh ek open-source software library hai jo primarily computer vision aur image processing ke liye use hoti hai.
## Features of OpenCV
1) Image Processing
2) Video Analysis
3) Face Detection
4) Object Detection
5) Machine Learning
## Uses
- self driving cars ke liye object detection
- facial recognition system
- augmented applications(AR)
- medical image processing
- surveliance system
  
What i understand is learning OpenCV with `python` is really easy 😘

---
# What kind of different things we can do with OpenCV Code
## Image Reading
```python
img=cv2.imread('vk2.jpg)
cv2.imshow("image",img)
cv2.waitKey(0)
```
## For Saving Image Again
```python
img=cv2.imread("vk2.jpg")
cv2.imwrite("savenewimage",img)
cv2.imshow("original image",img)
cv2.waitKey(0)
```
## For Resizing the image
```python
img=cv2.imread("vk2.jpg")
rsize=cv2.resize(img,(200,200),interolation=cv2.INTER_AREA)
cv2.imshow("image",rsize)
cv2.waitKey(0)
```
## For Roatating an image from center 
```python
img=cv2.imread("vk2.jpg")
width=img.shape[0]
height=img.shape[1]
centre=(height/2,width/2)
m=cv2.getRotationMatrix2D(centre,30,1)
img1=cv2.wrapAffine(img,m,(height,width))

cv2.imshow("roated image",img1)
cv2.imshow("original image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## For bluring an image
```python
img=cv2.imread("vk2.jpg")
gblur=cv2.GaussianBlur(img,(111,111),0)
cv2,imshow("real image",img)
cv2.imgshow("blur image",gblur)
cv2.waitKey(0)
cv2.DestroyAllWindows()
```
## For making border in immage
```python
import cv2
img=cv2.imread("vk2.jpg")
border=cv2.copyMakeBorder(img,20,20,20,20,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("border image",border)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## For changing image color
```python
import cv2
img=cv2.imread("vk2.jpg")
img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("originali image",img)
cv2.imshow("new image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## For making circle,rectangle in image and also for writing in image
```python
img=cv2.imread("vk2.jpg")
line=cv2.line(img,(0,0),(500,700),(255),10)
rectangle=cv2.rectangle(img,(0,0),(500,700),255,10)
circle=cv2.circle(img,(250,350),150,255,thickness=10)
cv2.imshow("image",circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
---
# Small Project : Circle Detection
starting mein thoda kam samajh aayega shyd , but agar try karoge to aajayega ...
```python
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
```
---
Thankyou for reading :)
In code you will read `vk2.jpg` everywhere ,but i wanted to tell Vk is none other than my inspiration Sir Virat Kohli ..

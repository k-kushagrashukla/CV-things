# Air Canvas
 A virtual drawing tool that allows you to draw in the air using hand gestures.
 
 ![Screenshot 2025-01-31 231601](https://github.com/user-attachments/assets/da694dd0-ceb1-4f44-8929-5cc1ef409bd6)

 
## Highlights
-Used OpenCV to capture live video from webcam

-Used MediaPipe for hand tracking and for gesture detection

-Used Deque for fast and efficient data storing

## Explanation (line by line explanation of every code)

```python
import cv2
import numpy as np
import mediapipe as mp
from collection import deque
```
cv2: Open library, image processing ke liye.

numpy: mathematical operation aur array ke liye

mediapipe: handtracking aur landmarks ko detect karne ke liye

deque: data structure jo painting ke points ko store karega

---

```python
bpoints=[deque(maxlen=1024)]
gpoints=[deque(maxlen=1024)]
rpoints=[deque(maxlen=1024)]
ypoints=[deque(maxlen=1024)]
```
har color ke liye ek deque banaya gya hai , taaki vo drawing points store kar sake

---

```python
colors=[(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex=0
```
RGB values ke through color define kiye gye hai, aur colorIndex selected color ko detect karega.

---

```python
paintWindow=np.zeros((471,636,3)) + 255
```
Ek blank white paint window banayi gyi hai (471*636 pixels)

---

```python
paintWindow=cv2.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)
paintWindow=cv2.rectangle(paintWindow,(160,1),(255,65),(255,0,0),2)
paintWindow=cv2.rectangle(paintWindow,(275,1),(370,65),(0,255,0),2)
paintWindow=cv2.rectangle(paintWindow,(390,1),(485,65),(0,0,255),2)
paintWindow=cv2.rectangle(paintWindow,(505,1),(600,65),(0,255,255),2)

cv2.putText(paintWindow,"CLEAR",(49,33),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow,"BLUE",(185,33),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow,"GREEN",(298,33),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow,"RED",(420,33),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow,"YELLOW",(520,33),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
```
Clear aur color buttons banaye gye hai , widnow top par

---

```python
cv2.namedWindow('paint',cv2.WINDOW_AUTOSIZE)
```
paintwindow ko display karne ke liye ek opencv window banayi gyi hai

---

```python
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils
```
sirf ek haath detect karne ke liye (max_num_hands=1)
minimum_detection_confidence 0.7 set ki gyi hai
landmarks draw karne ke liye mpdraw

---

```python
cap=cv2.VideoCapture(0)
ret,frame_temp=cap.read()
while ret:
    ret,frame=cap.read()
    x,y,c=frame.shape
```
Jab tak frame open hai tab tak , frame capture hota rahega .
frame ka size(x,y,c) store kiya gya hai 

---

```python
frame=cv2.flip(frame,1)
    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
```
frame ko horizontally filp karte hai, BGR(OpenCV format) se RGB(mediapipe ke liye) convert kiya

---

```python
frame=cv2.rectangle(frame,(40,1),(140,65),(0,0,0),2)
    frame=cv2.rectangle(frame,(160,1),(255,65),(255,0,0),2)
    frame=cv2.rectangle(frame,(275,1),(370,65),(0,255,0),2)
    frame=cv2.rectangle(frame,(390,1),(485,65),(0,0,255),2)
    frame=cv2.rectangle(frame,(505,1),(600,65),(0,255,255),2)
    cv2.putText(frame, "CLEAR",(49,33),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"BLUE",(185,33),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"GREEN",(298,33),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"RED",(420,33),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"YELLOW",(520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
```
same buttons (CLEAR,BLUE etc) live webcam par bhi draw karte hai 

---

```python
result=hands.process(framergb)
```
mediapipe model ko current frame provide karta hai taaki vo hand landmarks detect kare.

---

```python
if result.multi_hand_landmarks:
        landmarks=[]
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx=int(lm.x*640)
                lmy=int(lm.y*480)

                landmarks.append([lmx,lmy])

```
Agar `result_multi_hand_landmarks` empty nahi hai , iska matlab hai ki frame mein haath detect hogya hai aur ek khaali list banayi gyi hai jisme haath ke saare landmarks ke coordinate store honge.

Agar frame mein ek se zyada hand detect hote hai to dono hands ko process karega

har haath le 21 landmarks ko loop karke access karte hai 

![WhatsApp Image 2025-02-01 at 12 09 20_91db398a](https://github.com/user-attachments/assets/73ab6df3-84dc-4fec-9a40-b9f4a442ce3b)


`lm.x and lm.y` normalized value hoti hai (0 to 1 ke beech)

640 aur 480 screen ki height aur width hai 

aur fir unhe multiply karke screen ke pixels ke coordinates mein convert karte hai 

x aur y ko ek list mein store karte hai .

---

```python
fore_finger = (landmarks[8][0],landmarks[8][1])
        center= fore_finger
        thumb=(landmarks[4][0], landmarks[4][1])
```
index finger aur thumb ki positions nikalte hai

---

```python
if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index+=1
            gpoints.append(deque(maxlen=512))
            green_index+=1
            rpoints.append(deque(maxlen=512))
            red_index+=1
            ypoints.append(deque(maxlen=512))
            yellow_index+=1

```
agar thumb aur index finger ki distance kam ho to , ek naya list start hoti hai 

---

```python
elif center[1] <=65:
            if 40 <= center[0] <=140:
                bpoints=[deque(maxlen=512)]
                gpoints=[deque(maxlen=512)]
                rpoints=[deque(maxlen=512)]
                ypoints=[deque(maxlen=512)]

                blue_index=0
                green_index=0
                red_index=0
                yellow_index=0

                paintWindow[67:,:,:] = 255
            elif 160 <=center[0]<=255:
                colorIndex=0
            elif 275 <=center[0] <=370:
                colorIndex=1
            elif 390 <=center[0] <=485:
                colorIndex=2
            elif 505 <= center[0] <=600:
                colorIndex=3
```
Agar index finger btton area mein hai :
. clear button press hone par saare points reset hojaaayenge
. aur alag alag colors choose karne par indexcolor value change hogi

---

```python
else:
            if colorIndex ==0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex ==1:
                gpoints[green_index].appendleft(center)
            elif colorIndex ==2:
                rpoints[red_index].appendleft(center)
            elif colorIndex ==3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index+=1
        gpoints.append(deque(maxlen=512))
        green_index+=1
        rpoints.append(deque(maxlen=512))
        red_index+=1
        ypoints.append(deque(maxlen=512))
        yellow_index+=1
    
```
Agar finger, button area ke bahar hai toh selected color ke according list mein point add karte hai.

---

```python
oints=[bpoints,gpoints,rpoints,ypoints]

    for i in range(len(points)):
        for j in range(1,len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(paintWindow,points[i][j][k-1],points[i][j][k],colors[i],2)
```
Har color ke stroke ko frame aur paintwindow dono pe draw karte hai 

---

```python
cv2.imshow("Output",frame)
    cv2.imshow("Paint",paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break
```
webcam aur paintwindow ko display karte hai 

---
Thankyou for reading this , if you liked it then pls drop a 🌟:)


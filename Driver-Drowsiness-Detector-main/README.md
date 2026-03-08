# Driver Drowsiness detector
Ever wished to prevent accidents before they happen ? That's what i have created Driver Drwsiness detctor. Basically a safety tool  that detects driver fatigue in real time!
and hence this system helps in road safety

![Screenshot 2025-04-09 141938](https://github.com/user-attachments/assets/161c273e-92ee-4301-9150-73288f70509a)

![Screenshot 2025-04-09 141955](https://github.com/user-attachments/assets/015fd34a-9661-4d1a-8f80-fa5508ca46c0)



## Algorithms highlight
- Used OpenCV library for image processing
  
- Used numpy for mathematical operation and for array manipulation
  
- Used dlib for face detection and for face landmarks

- Used face_utils to convert face landmarks to numpy array.

## Code Explanation
```python
import cv2
import numpy as np
import dlib
from imutils import face_utils
```
cv2: Used for image processing

numpy : Used for mathematical operation and for array manipulation

dlib: Used for face detection and for face landmarks

face_utils : Used to convert face landmarks to numpy array

```python
cap = cv2.VideoCapture(0)
```
Yeh line webcam ko on karti hai taaki real time video li jaa ske

```python
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```
`detctor` : yeh face ko detct karne ka kaam karta hai

`predictor` : yeh 68 face landmarks detect karta hai ( iske liye hame ek pretrained model chahiye hota hai )

```python
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
```
yahan variables track karte hai ki user sleepy hai ya drowsy ya fir active :)

`color` : yeh text ka color define karta hai

```python
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)
```
yeh function do points ke beech ki distance find karta hai.

```python
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    
    if ratio > 0.25:
        return 2  # Open eyes
    elif 0.21 <= ratio <= 0.25:
        return 1  # Partially closed
    else:
        return 0  # Closed
```
yeh function eyes ka blink ratio find karta hai 

agar `ratio>0.25` then eyes are open

agar `0.21<=ratio <=0.25` then eyes are partially closed 

aur agar `value 0 ` hai to fir eyes close hai 

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
```
- video capture karega webcam frame by frame

- agar video capture nhi hoti to frame break hoga

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
```
frame ko graysale mein convert karte hai , kyoki tabhi vo better perform karta hai

`faces`: ye faces ko detct karta hai

```python
for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```
yeh face ke around ek green box draw karega 

```python
landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
```
yeh code face ke 68 landmarks ko detect karega aur unhe numpy array mein convert karega

```python
left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
```
yeh code dono aankho ka blink status check karega

```python
 if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
```
- Agar aankhein band hain (0 value) to sleep count badhta hai.
- Agar yeh 6 se zyada frames tak chalta raha to status `"SLEEPING"` ho jayega.

```python
 elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
```
- Agar aankhein aadhi band hain (1 value) to drowsy count badhta hai.
- Agar yeh 6 se zyada frames tak chalta raha to status `"Drowsy"` ho jayega.

```python
else:
     sleep = 0
     drowsy = 0
     active += 1
     if active > 6:
       status = "Active :)"
        color = (0, 255, 0)
```
- Agar aankhein poori tarah khuli hain (2 value) to active count badhta hai.
- Agar yeh 6 frames tak chalta raha to status `"Active :)"` ho jayega.

```python
 cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    
    cv2.imshow("Frame", frame)
```
- yeh status screen pe draw karega.
- aur face [e dots yaani landmarks draw karega.
- yeh processed frame ko show karta hai

```python
if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break
```
in last `Esc` click karne par loop break hojaayega 

---
If you like explanation , then please drop a 🌟 :)


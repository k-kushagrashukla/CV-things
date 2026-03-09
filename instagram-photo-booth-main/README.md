## Built a Real-Time Instagram-Style Photo Booth with OpenCV + Python!

 - Harnessed the power of OpenCV and cvzone to create a live webcam application that detects faces and overlays fun filters like 👓 glasses, 🎩 hats, 🐶 dog faces, and even 🧛 vampire teeth — all in real-time!

## Tech Highlights:
✅Used FaceMesh & Face Detection modules for precise landmark tracking.

✅Applied PNG overlays smartly with accurate alignment (eyes, nose, chin).

✅Supports dynamic switching between multiple filters (keys 1–8).

✅Resized and placed filters using real facial proportions (no more mustache-on-neck bugs!).

- This project demonstrates how computer vision can be used to build fun, interactive, and engaging user experiences — a stepping stone toward augmented reality apps!
---
## Code Explanation

```python
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.Utils import overlayPNG
import os
```
- cv2: OpenCV, image aur video processing ke liye.

- cvzone: OpenCV ka wrapper, jo face detection, overlay, etc. easy banaata hai.

- FaceDetector: Face detect karne ke liye.

- FaceMeshDetector: Face ke detailed landmarks (jaise eyes, nose, mouth) detect karne ke liye.

- overlayPNG: PNG image ko transparent background ke sath overlay karne ke liye.

- os: File path se deal karne ke liye (used for folders).

```pythom
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
```
- VideoCapture(0): Webcam ko start karta hai.

- set(3, 1280) aur set(4, 720): Frame width aur height set karte hain (HD resolution).

```python
filter_path = "Filters"
filters = {
    '1': cv2.imread(...),
    ...
}
```
-"Filters" folder mein se PNG filters load kiye ja rahe hain (glasses, hat, mustache, etc.)

- Har filter ek key (1–7) se mapped hai.

```python
current_filter = '1'  # default glasses

faceDetector = FaceDetector(minDetectionCon=0.6)
meshDetector = FaceMeshDetector(staticMode=False, maxFaces=1)
```
- Default filter glasses hai.

- FaceDetector: Face detect karta hai.

- FaceMeshDetector: Face ke chhote points (landmarks) detect karta hai jaise aankhon ke corners, naak ka tip, etc.

```python
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
```
- Webcam se frame read hota hai.

- flip(img, 1) – Mirror effect dene ke liye (left-right ulta).

```python
img, bboxs = faceDetector.findFaces(img, draw=False)
img, faces = meshDetector.findFaceMesh(img, draw=False)
```
- bboxs: Face ka bounding box deta hai (x, y, width, height).

- faces: Landmarks (468 points) list deta hai.

```python
if bboxs and faces:
    ...
    filter_img = filters.get(current_filter)
    ...
```
- Agar face milta hai tab hi filter apply karo.

- Har filter ke liye alag-alag position calculation ki ja rahi hai

```python
eye_center = (left eye + right eye) / 2
eye_width = distance between eyes
resize + overlay on eyes
```
- for hat and crown : Place filter above forehead using bbox x, y, w, h

```python
cv2.putText(img, "Filter: ...", ...)
```
- Real-time screen par current filter aur controls show karta hai.

```python
key = cv2.waitKey(1)
if key != -1:
    ...
```
- Agar user koi key press kare to:

- 'q': Quit karo.
'1' to '7': Filter change karo.







![Screenshot 2025-04-05 230719](https://github.com/user-attachments/assets/54ce713d-54aa-43d0-afd6-76b9e2377ba7)

![Screenshot 2025-04-07 111411](https://github.com/user-attachments/assets/52846718-bada-411f-9abc-943bbf2f64be)

# Balloon Pop Game Using OpenCV
Simple Balloon Pop Game using OpenCV and MediaPipe hand tracking. The goal is to pop the balloon by touching it with your fingertip.
## Algorithm Highlight
- Used OpenCv for image processing.
- Used numpy for numerical operations.
- Used mediapipe for hand tracking and detection.
- Used random for generating random position of balloons.
## Code Explanation
```python
import cv2
import numpy as np
import mediapipe as mp
import random
```
importing all necessary modules to make this real-time balloon pop game.

```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
```
- Mediapipe are initializing the hand
- taking hand tracking accuracy as 70%.

```python
WIDTH, HEIGHT = 640, 480

balloon_radius = 30
balloon_x = random.randint(balloon_radius, WIDTH - balloon_radius)  
balloon_y = HEIGHT - balloon_radius  # Start position neeche se
balloon_speed = 5  # Balloon ki speed upar jaane ki
score = 0  # Initial score
```
- Ballon size (radius) is fixed.
- Balloon X-coordinate is random to generate new position of balloon every time.
- Balloon speed is fix.
- Score of game is initially zero.

```python
def detect_hands(frame):
    global balloon_x, balloon_y, score
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = hands.process(frame_rgb)  # Detect hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            
            # Check if finger is inside the balloon
            if (x - balloon_x) ** 2 + (y - balloon_y) ** 2 <= balloon_radius ** 2:
                balloon_x = random.randint(balloon_radius, WIDTH - balloon_radius)  # Naya X position
                balloon_y = HEIGHT - balloon_radius  # Balloon reset
                score += 1  # Score increase

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

```
- Converting frame to RGB because mediapipe only works on that
- Detecting hand
- Getting index finger position (means X,Y coordinate)
- If finger is inside the balloon , then balloon is going to pop and score is going to increase.
- Then only new balloon get generated.
- Drawing hand landmarks  

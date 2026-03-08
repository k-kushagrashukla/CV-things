import cv2
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[:2]

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if len(lm_list) >= 8:  # Ensure at least index and thumb are detected
                x1, y1 = lm_list[4]   # Thumb tip
                x2, y2 = lm_list[8]   # Index finger tip
                cv2.circle(img, (x1, y1), 8, (255, 0, 0), -1)
                cv2.circle(img, (x2, y2), 8, (255, 0, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Calculate distance between thumb and index finger
                length = math.hypot(x2 - x1, y2 - y1)

                # Convert distance to volume range
                vol = np.interp(length, [20, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                # Display volume level
                vol_bar = np.interp(vol, [min_vol, max_vol], [400, 150])
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'Vol: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))}%', 
                            (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

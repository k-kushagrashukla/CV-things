import cv2
import mediapipe as mp
import pyautogui
import time

# Setup
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
action_cooldown = 0.3  # Reduced for faster response
last_action_time = time.time()

def count_open_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    height, width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            cx = int(hand.landmark[9].x * width)
            cy = int(hand.landmark[9].y * height)

            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

            if time.time() - last_action_time > action_cooldown:
                # Fist detection (0 or 1 fingers up)
                fingers_up = count_open_fingers(hand)
                if fingers_up <= 1:
                    pyautogui.press("down")
                    cv2.putText(frame, "ROLL (Fist)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    last_action_time = time.time()
                else:
                    dx = cx - prev_x
                    dy = cy - prev_y

                    if dx < -30:
                        pyautogui.press("left")
                        cv2.putText(frame, "LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        last_action_time = time.time()
                    elif dx > 30:
                        pyautogui.press("right")
                        cv2.putText(frame, "RIGHT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        last_action_time = time.time()
                    elif dy < -30:
                        pyautogui.press("up")
                        cv2.putText(frame, "JUMP", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        last_action_time = time.time()

            prev_x, prev_y = cx, cy

    cv2.imshow("Fast Subway Surfer Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

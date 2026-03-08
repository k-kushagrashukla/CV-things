import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Hand tracking module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Ball properties
ball_x, ball_y = 320, 240
ball_dx, ball_dy = 4, 4
ball_radius = 10

# Paddle properties
paddle_width = 100
paddle_height = 20
paddle_x = 270
paddle_y = 450

score = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand position (Index Finger Tip)
            index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 640)
            paddle_x = index_x - paddle_width // 2  # Align paddle with finger

    # Ball movement
    ball_x += ball_dx
    ball_y += ball_dy

    # Collision with walls
    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= 640:
        ball_dx = -ball_dx

    if ball_y - ball_radius <= 0:
        ball_dy = -ball_dy

    # Collision with paddle
    if paddle_y <= ball_y + ball_radius <= paddle_y + paddle_height and paddle_x <= ball_x <= paddle_x + paddle_width:
        ball_dy = -ball_dy
        score += 1

    # Game over
    if ball_y + ball_radius >= 480:
        ball_x, ball_y = 320, 240
        ball_dx, ball_dy = 4, 4
        score = 0

    # Draw paddle and ball
    cv2.rectangle(frame, (paddle_x, paddle_y), (paddle_x + paddle_width, paddle_y + paddle_height), (0, 255, 0), -1)
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Score display
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Ping Pong", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

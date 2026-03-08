import cv2
import numpy as np
import mediapipe as mp
import random

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Game window size
WIDTH, HEIGHT = 640, 480

# Balloon properties
balloon_radius = 30
balloon_x = random.randint(balloon_radius, WIDTH - balloon_radius)
balloon_y = HEIGHT - balloon_radius
balloon_speed = 5
score = 0

def detect_hands(frame):
    global balloon_x, balloon_y, score
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            
            # Check if the fingertip is inside the balloon (pop it)
            if (x - balloon_x) ** 2 + (y - balloon_y) ** 2 <= balloon_radius ** 2:
                balloon_x = random.randint(balloon_radius, WIDTH - balloon_radius)
                balloon_y = HEIGHT - balloon_radius
                score += 1
                
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def draw_objects(frame):
    global balloon_y
    
    # Draw balloon
    cv2.circle(frame, (balloon_x, balloon_y), balloon_radius, (0, 0, 255), -1)
    
    # Draw score
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def update_balloon():
    global balloon_y
    
    balloon_y -= balloon_speed
    
    # Reset balloon if it reaches the top
    if balloon_y < 0:
        balloon_y = HEIGHT - balloon_radius

def main():
    global score
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        detect_hands(frame)
        draw_objects(frame)
        update_balloon()
        
        cv2.imshow("Balloon Pop Game", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

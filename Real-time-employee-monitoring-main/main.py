import cv2
import time
import numpy as np
from trackers.presence_tracker import PresenceTracker
from trackers.eye_tracker import EyeTracker
from trackers.lip_movement import LipMovementDetector
from utils.logging_utils import setup_logger, log_session
from utils.drawing_utils import draw_face_annotations


LOG_FILE = 'data/presence_log.csv'
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIP_POINTS = [13, 14, 78, 308]

setup_logger(LOG_FILE)

presence_tracker = PresenceTracker()
eye_tracker = EyeTracker()
lip_tracker = LipMovementDetector()

cap = cv2.VideoCapture(0)

present = False
session_start_time = None
last_presence_time = time.time()
total_present_seconds = 0
eye_closure_start = 0
continuous_eye_closed = False
talking_state_changes = []  

EYE_CLOSURE_THRESHOLD = 300  
ABSENCE_THRESHOLD = 120      
TALKING_FREQUENCY_THRESHOLD = 4  
MONITORING_WINDOW = 60      

last_status = None
last_log_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame = cv2.flip(frame, 1)


    present_now = presence_tracker.detect_presence(frame)
    eye_status, ear = eye_tracker.detect_eye_state(frame)
    mouth_status, mar, state_change = lip_tracker.detect_talking(frame)  # Get state change flag
    
   
    if state_change:
        talking_state_changes.append(current_time)
        print(f"Talking state change detected at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
    

    talking_state_changes = [t for t in talking_state_changes if current_time - t <= MONITORING_WINDOW]
    talking_frequency = len(talking_state_changes)


    if eye_status == "Eyes Closed":
        if not continuous_eye_closed:
            eye_closure_start = current_time
            continuous_eye_closed = True
        eye_closure_duration = current_time - eye_closure_start
    else:
        continuous_eye_closed = False
        eye_closure_duration = 0

 
    if present_now:
        if not present:
            present = True
            session_start_time = current_time
        last_presence_time = current_time
        color = (0, 255, 0)
        display_status = "Present"
    else:
        present = False
        color = (0, 0, 255)
        display_status = "Absent"


    eye_issue = eye_closure_duration > EYE_CLOSURE_THRESHOLD
    talking_issue = talking_frequency > TALKING_FREQUENCY_THRESHOLD
    absence_issue = not present_now and (current_time - last_presence_time > ABSENCE_THRESHOLD)

    working_status = "Not Working " if (eye_issue or talking_issue or absence_issue) else "Working "

   
    if working_status != last_status and (current_time - last_log_time > 5):
        if session_start_time:
            duration = current_time - session_start_time
            log_session(
                LOG_FILE,
                session_start_time,
                current_time,
                duration,
                working_status
            )
            last_log_time = current_time
            print(f"Logged status change: {working_status} at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        last_status = working_status
        session_start_time = current_time  

    if eye_status != "Unknown":
        mesh_results = eye_tracker.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                draw_face_annotations(frame, face_landmarks, [LEFT_EYE, RIGHT_EYE], LIP_POINTS)


    cv2.putText(frame, f"Presence: {display_status}", (30, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Eye: {eye_status}", (30, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Mouth: {mouth_status}", (30, 115), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    cv2.putText(frame, f"Talk Events: {talking_frequency}/{TALKING_FREQUENCY_THRESHOLD}", (30, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 100, 255), 2)
    cv2.putText(frame, f"Status: {working_status}", (30, 185), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    

    if continuous_eye_closed:
        cv2.putText(frame, f"Eyes Closed: {int(eye_closure_duration)}s/{EYE_CLOSURE_THRESHOLD}s", 
                   (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
   
    if not present_now and present:
        absence_time = current_time - last_presence_time
        cv2.putText(frame, f"Absent: {int(absence_time)}s/{ABSENCE_THRESHOLD}s", 
                   (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    

    cv2.namedWindow("Employee Monitor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Employee Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Employee Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if session_start_time:
            duration = current_time - session_start_time
            log_session(
                LOG_FILE,
                session_start_time,
                current_time,
                duration,
                working_status
            )
        break

cap.release()
cv2.destroyAllWindows()
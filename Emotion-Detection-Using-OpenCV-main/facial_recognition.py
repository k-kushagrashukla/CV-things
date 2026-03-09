#cv2: OpenCV library face detection aur video capture ke liye use hoti hai.
#DeepFace: Emotion detection ke liye pre-trained models ko access karta hai.
import cv2
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#cv2.VideoCapture(0) webcam se live video capture karta hai.
#0 ka matlab hai default webcam.

cap = cv2.VideoCapture(0)

while True:
    # cap.read():
    #Frame-by-frame webcam se video capture karta hai.
    #ret: Video capture sahi ho raha hai ya nahi, isko check karne ke liye.
    #frame: Ek single frame ka data store karta hai.
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Grayscale conversion se processing fast hoti hai. Haar cascade model grayscale images pe kaam karta hai.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor: Har step pe image ko kitna scale down karna hai.
    #minNeighbors: Har face detection ko validate karne ke liye neighbors.
    #minSize: Minimum size jo ek face ke liye consider kiya jaaye.
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Analyze emotions using DeepFace
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Handle the result depending on its format
            if isinstance(results, list):  # Newer versions return a list
                analysis = results[0]  # Extract the first dictionary
            else:
                analysis = results  # Older versions return a dictionary directly

            # Get the dominant emotion
            dominant_emotion = analysis['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error during emotion analysis: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Pressing 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

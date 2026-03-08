import cv2
import mediapipe as mp
from deepface import DeepFace


image_path = "group photo.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Resize for better viewing (optional)
scale = 0.5
display_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run face detection
    results = face_detection.process(image_rgb)

    if not results.detections:
        print("No faces detected.")
    else:
        for detection in results.detections:
            # Extract bounding box (MediaPipe gives relative values)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Ensure with bounds
            x, y = max(0, x), max(0, y)
            face_roi = image[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                analysis = result[0] if isinstance(result, list) else result
                emotion = analysis['dominant_emotion']

                # Draw on display image
                sx, sy, sw, sh = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
                cv2.rectangle(display_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.putText(display_img, emotion, (sx, sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Emotion detection error: {e}")


cv2.imshow("emotion detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

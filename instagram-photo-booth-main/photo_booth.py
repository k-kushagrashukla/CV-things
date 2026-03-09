import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.Utils import overlayPNG
import os

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load Filters (Make sure these exist in the 'Filters' folder)
filter_path = "Filters"
filters = {
    '1': cv2.imread(f"{filter_path}/glasses.png", cv2.IMREAD_UNCHANGED),
    '2': cv2.imread(f"{filter_path}/hat.png", cv2.IMREAD_UNCHANGED),
    '3': cv2.imread(f"{filter_path}/mustache.png", cv2.IMREAD_UNCHANGED),
    '4': cv2.imread(f"{filter_path}/dog_face.png", cv2.IMREAD_UNCHANGED),
    '5': cv2.imread(f"{filter_path}/vampire_teeth.png", cv2.IMREAD_UNCHANGED),
    '6': cv2.imread(f"{filter_path}/crown.png", cv2.IMREAD_UNCHANGED),
    '7': cv2.imread(f"{filter_path}/cool_shades.png", cv2.IMREAD_UNCHANGED),
    
}

# Default Filter Key
current_filter = '1'

# Initialize Detectors
faceDetector = FaceDetector(minDetectionCon=0.6)
meshDetector = FaceMeshDetector(staticMode=False, maxFaces=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, bboxs = faceDetector.findFaces(img, draw=False)
    img, faces = meshDetector.findFaceMesh(img, draw=False)

    if bboxs and faces:
        bbox = bboxs[0]['bbox']
        x, y, w, h = bbox
        lm = faces[0]  # Landmark list

        # Load selected filter
        filter_img = filters.get(current_filter)
        if filter_img is not None:
            # Position based on filter type
            if current_filter == '1' or current_filter == '7':  # Glasses / Cool Shades
                eye_center_x = (lm[33][0] + lm[263][0]) // 2
                eye_center_y = (lm[33][1] + lm[263][1]) // 2
                eye_width = abs(lm[263][0] - lm[33][0]) * 2
                filter_resized = cv2.resize(filter_img, (int(eye_width), int(eye_width * filter_img.shape[0] / filter_img.shape[1])))
                img = overlayPNG(img, filter_resized, [eye_center_x - filter_resized.shape[1] // 2, eye_center_y - filter_resized.shape[0] // 2])

            elif current_filter == '2' or current_filter == '6':  # Hat / Crown
                hat_width = w + 40
                hat_height = int(hat_width * filter_img.shape[0] / filter_img.shape[1])
                filter_resized = cv2.resize(filter_img, (hat_width, hat_height))
                img = overlayPNG(img, filter_resized, [x - 20, y - hat_height + 10])

            elif current_filter == '3':  # Mustache
                nose_bottom = lm[2]  # Use nose tip for better positioning
                mustache_width = int(w * 0.5)
                filter_resized = cv2.resize(filter_img, (mustache_width, int(mustache_width * filter_img.shape[0] / filter_img.shape[1])))
                img = overlayPNG(img, filter_resized, [nose_bottom[0] - filter_resized.shape[1] // 2, nose_bottom[1]])

            elif current_filter == '5':  # Vampire Teeth
                mouth_x = (lm[13][0] + lm[14][0]) // 2
                mouth_y = (lm[13][1] + lm[14][1]) // 2
                filter_width = int(w * 0.5)
                filter_resized = cv2.resize(filter_img, (filter_width, int(filter_width * filter_img.shape[0] / filter_img.shape[1])))
                img = overlayPNG(img, filter_resized, [mouth_x - filter_resized.shape[1] // 2, mouth_y - 5])

            elif current_filter == '4':  # Dog Face (centered on whole face)
                face_size = int(w * 1.1)
                filter_resized = cv2.resize(filter_img, (face_size, int(face_size * filter_img.shape[0] / filter_img.shape[1])))
                img = overlayPNG(img, filter_resized, [x - 10, y - 20])

            

    # Show instructions
    cv2.putText(img, f"Filter: {current_filter} | Press 1-7 to switch filters | Q to quit", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display
    cv2.imshow("Photo Booth", img)

    key = cv2.waitKey(1)
    if key != -1:
        try:
            key_char = chr(key)
            if key_char == 'q':
                break
            elif key_char in filters:
                current_filter = key_char
        except:
            pass

cap.release()
cv2.destroyAllWindows()


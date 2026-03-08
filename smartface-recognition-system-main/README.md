# "SmartFace Recognition System" 🚀
This is a real-time face recognition system using OpenCV and face_recognition library. It identifies pre-registered faces using a webcam and displays their names on the screen. If a detected face is not recognized, it is labeled as "Unknown".

## Key features
✅ Live Face Detection & Recognition 📷

✅ Fast Processing with Image Resizing ⚡

✅ Supports Multiple Faces in a Single Frame 👨‍👩‍👧‍👦

✅ Easy to Add New Faces 🖼️

✅ Scalable & Efficient for Larger Datasets 📂

## How It Works
1️⃣ Load Known Faces

The program starts by registering known faces using images stored on disk.

Each face is encoded and saved in known_face_encodings with corresponding names in known_face_names.

2️⃣ Capture Live Video

A webcam feed is initiated using cv2.VideoCapture(0).

3️⃣ Detect & Recognize Faces

The incoming video frames are resized to 50% to speed up face detection.

The program detects faces and extracts encodings.

These encodings are compared with stored faces using the face_recognition library.

4️⃣ Display Recognized Faces

If a match is found, the name of the person is displayed on the screen.

If no match is found, the label "Unknown" is assigned.

5️⃣ Live Feed Display & User Exit

The processed frame is displayed in a window, and the user can exit by pressing 'Q'.

## Possible Use Cases
🔹 Attendance Systems 🏫

🔹 Secure Access Control 🔒

🔹 Smart Home Systems 🏠

🔹 AI-Powered Surveillance 🎥

🔹 Interactive Kiosks & Event Management 🏟️


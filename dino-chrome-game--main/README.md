# Gesture-Controlled Chrome Dino Game 🦖🖐️

Super excited to share a fun project I recently built — a real-time hand gesture-controlled Chrome Dino game! No keyboard needed — just raise a finger to make Dino jump! 🖐️➡️🕹️

![Screenshot 2025-04-13 190733](https://github.com/user-attachments/assets/a286ddae-f3b6-497d-99b3-15cdfc1c6c75)


## Tech Stack:
- OpenCV: Captures live webcam video.
- MediaPipe: Tracks and detects hand landmarks.
- PyAutoGUI: Simulates keyboard input (space bar) to make Dino jump.
- Python: The glue that binds it all together.

## 🧠 How It Works:
- The webcam feed is analyzed in real-time using MediaPipe's Hand module.
- A custom logic counts raised fingers.
- When exactly 1 finger is raised, the program sends a space key press to trigger a jump.
- A cooldown timer avoids repeated jumps from a single gesture.

## ✨ Features:
- Real-time hand tracking 🖐️
- Smooth, responsive interaction with a cooldown mechanism to reduce lag

## 🖐️gesture to play :
- 👆1 finger and dino jumps
- ✌️2 fingers and dino stays on ground

💡 This project was a fun way to explore computer vision, real-time input systems, and automation :)


# Eyeblink Detection Using OpenCV
Bascially , it is a detection system where we detect whether a eye is blinked or not ??
## We use these predefined functions for eye blink
```python
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
```
---
I made this as a game , where when webcam detect our face , so it will say press `S` to start the game ...........
the game is going to start then and the moment you blink your eye . the game will going to stop.
and again it will ask you to play the game , by pressing `S`.
But for closing the game/window you need to press `Q`

## When Face Is Not Detected
![Screenshot 2025-01-21 223316](https://github.com/user-attachments/assets/1662df12-bd7e-4b22-9958-3318e2b56cdb)
---

## When Eyes Get Detected
It'll say print `S`
(Ab yahan par main , apna khoobsurat chehra nhi dikhaunga , aank maarte hue 😉, Varna samaj kya sochega mere baare mein 🫣)

---


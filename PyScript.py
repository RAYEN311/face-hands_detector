import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands()
root = tk.Tk()
root.title("ben brahim hand_cap Tracker")
label = tk.Label(root)
label.pack()
smile_counter = 0
emotion_background = tk.Frame(root, bg='white')
emotion_background.pack(fill=tk.BOTH, expand=True)
prev_hand_landmarks = None

def detect_emotion(face_detections):
    global smile_counter
    for detection in face_detections:
        if detection.score[0] > 0.6: 
            smile_counter += 1
        else:
            smile_counter = max(0, smile_counter - 1)

    if smile_counter >= 10:
        emotion_background.config(bg='green')
    else:
        emotion_background.config(bg='red')

def capture_screenshot(frame):
    cv2.imwrite("my_pic_xd.png", frame)
    print("Screenshot saved as 'my_pic_xd.png'")

def update():
    global prev_hand_landmarks
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            detect_emotion(face_results.detections)
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                if prev_hand_landmarks is not None:
                    hand_movement = any(
                        abs(landmark.x - prev_landmark.x) + abs(landmark.y - prev_landmark.y) > 0.02
                        for landmark, prev_landmark in zip(landmarks.landmark, prev_hand_landmarks.landmark)
                    )
                    if hand_movement:
                        capture_screenshot(frame)

                prev_hand_landmarks = landmarks
                for idx, landmark in enumerate(landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        label.img = img_tk
        label.config(image=img_tk)
        root.after(10, update)

update()  
root.mainloop()


cap.release()
cv2.destroyAllWindows()

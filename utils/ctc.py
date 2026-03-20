import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Utility Functions ----------------

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def finger_is_extended(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

# ---------------- Gesture Detection ----------------

def detect_gesture(lm):
    if not lm or len(lm) < 21:
        return "No Hand"

    index = finger_is_extended(lm, 8, 6)
    middle = finger_is_extended(lm, 12, 10)
    ring = finger_is_extended(lm, 16, 14)
    pinky = finger_is_extended(lm, 20, 18)

    thumb_up = lm[4][1] < lm[3][1]
    thumb_down = lm[4][1] > lm[3][1]

    # Five
    if index and middle and ring and pinky:
        return "Five"

    # Fist
    if not index and not middle and not ring and not pinky:
        return "Fist"

    # One
    if index and not middle and not ring and not pinky:
        return "One"

    # Two
    if index and middle and not ring and not pinky:
        return "Two"

    # Three
    if index and middle and ring and not pinky:
        return "Three"

    # Rock
    if index and not middle and not ring and pinky:
        return "Rock"

    # Pinch
    if calculate_distance(lm[4], lm[8]) < 40:
        return "Pinch"

    # OK
    if calculate_distance(lm[4], lm[8]) < 40 and middle and ring and pinky:
        return "OK"

    # Thumbs Up
    if thumb_up and not index and not middle:
        return "Thumbs Up"

    # Thumbs Down
    if thumb_down and not index and not middle:
        return "Thumbs Down"

    # Grab
    center = lm[0]
    if all(calculate_distance(lm[i], center) < 60 for i in [8,12,16,20]):
        return "Grab"

    return "Unknown"

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    gesture = "No Hand"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            lm = []
            for landmark in hand_landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                lm.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

            gesture = detect_gesture(lm)

    cv2.putText(frame, gesture, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("Hand Gesture System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
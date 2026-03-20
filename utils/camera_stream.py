import cv2
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
splits = ["train", "val"]
gestures = ["point", "click"]

for s in splits:
    for g in gestures:
        os.makedirs(os.path.join(DATASET_DIR, s, g), exist_ok=True)

cap = cv2.VideoCapture(0)

current_split = "train"
current_gesture = "point"


def get_next_index(filepath):
    files = [f for f in os.listdir(filepath) if f.endswith(".png")]
    if not files:
        return 0
    return max(int(f.split(".")[0]) for f in files)


counter = {s: {g: 0 for g in gestures} for s in splits}
counter[s][g] = get_next_index(os.path.join(DATASET_DIR, s, g))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    text = f"{current_split} | {current_gesture} | {counter[current_split][current_gesture]}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    try:
        if cv2.getWindowProperty("Dataset Capture", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

    if key == ord('q'):
        break
    elif key == ord('t'):
        current_split = "train"
    elif key == ord('v'):
        current_split = "val"
    elif key == ord('1'):
        current_gesture = "point"
    elif key == ord('2'):
        current_gesture = "click"
    elif key == ord('s'):
        counter[current_split][current_gesture] += 1
        filename = f"{counter[current_split][current_gesture]:04d}.png"
        path = os.path.join(DATASET_DIR, current_split, current_gesture, filename)
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()
cv2.destroyAllWindows()

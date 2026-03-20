import os
import random
import cv2
import torch
import numpy as np
from models.hrnet_w18 import LightweightHRNet

DEVICE = "cpu"
CHECKPOINT = "checkpoints/best.pth"
IMAGE_FOLDER = "dataset/FreiHAND/training/rgb"
IMAGE_SIZE = 224
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 21


def load_model():
    model = LightweightHRNet(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def preprocess_image(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = image.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.tensor(img).unsqueeze(0).to(DEVICE)


def heatmap_to_keypoints(heatmaps):
    keypoints = []
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))
    return keypoints


def draw_keypoints(image, keypoints):
    scale = IMAGE_SIZE / HEATMAP_SIZE
    for x, y in keypoints:
        px = int(x * scale)
        py = int(y * scale)
        cv2.circle(image, (px, py), 4, (0, 255, 0), -1)
    return image


def main():
    model = load_model()
    image_files = os.listdir(IMAGE_FOLDER)

    while True:
        file_name = random.choice(image_files)
        path = os.path.join(IMAGE_FOLDER, file_name)
        image = cv2.imread(path)

        if image is None:
            continue

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            heatmaps = torch.sigmoid(model(input_tensor))

        heatmaps = heatmaps[0].cpu().numpy()
        keypoints = heatmap_to_keypoints(heatmaps)

        display = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        output_image = draw_keypoints(display, keypoints)

        cv2.imshow("Prediction", output_image)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

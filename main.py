import random
from dataset_loader import DatasetLoader
import cv2
import numpy as np


def heatmap_to_keypoints(heatmaps, image_size=224, heatmap_size=64):

    keypoints = []

    scale = image_size / heatmap_size

    for heatmap in heatmaps:

        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        px = x * scale
        py = y * scale

        keypoints.append((px, py))

    return np.array(keypoints)


dataset = DatasetLoader(
    image_folder="dataset/FreiHAND/training/rgb",
    xyz_json="dataset/FreiHAND/training_xyz.json",
    k_json="dataset/FreiHAND/training_K.json",
    heatmap_size=64,
    image_size=224,
    augment=True
)

while True:

    index = random.randint(0, len(dataset)-1)

    image, heatmaps = dataset[index]

    image = image.permute(1, 2, 0).numpy() * 255
    image = image.astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    heatmaps = heatmaps.numpy()
    keypoints = heatmap_to_keypoints(heatmaps)

    for x, y in keypoints:
        cv2.circle(image_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imshow("keypoints", image_bgr)
    key = cv2.waitKey(0)

    if key == 27:
        break

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class DatasetLoader(Dataset):

    def __init__(
        self,
        image_folder,
        xyz_json,
        k_json,
        heatmap_size=64,
        image_size=224,
        augment=True
    ):

        self.image_folder = image_folder
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.augment = augment

        with open(xyz_json, "r") as f:
            self.xyz_data = json.load(f)

        with open(k_json, "r") as f:
            self.K_data = json.load(f)

        self.image_files = sorted(os.listdir(image_folder))

        self.xx, self.yy = np.meshgrid(
            np.arange(self.heatmap_size),
            np.arange(self.heatmap_size)
        )

    def __len__(self):
        return len(self.xyz_data)

    def project_to_2d(self, xyz, K):

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]

        uv = []

        for x, y, z in xyz:

            u = fx * x / z + cx
            v = fy * y / z + cy

            uv.append([u, v])

        return np.array(uv)

    def __getitem__(self, index):

        image_name = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            return self.__getitem__((index + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        xyz = np.array(self.xyz_data[index], dtype=np.float32).reshape(-1, 3)
        K = np.array(self.K_data[index], dtype=np.float32).reshape(3, 3)

        keypoints = self.project_to_2d(xyz, K)

        if self.augment:

            if np.random.rand() < 0.5:

                image = cv2.flip(image, 1)
                keypoints[:, 0] = self.image_size - keypoints[:, 0]

            angle = np.random.uniform(-25, 25)
            height, width = image.shape[:2]

            rot_mat = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                angle,
                1.0
            )

            image = cv2.warpAffine(image, rot_mat, (width, height))

            kp_homog = np.hstack(
                [keypoints, np.ones((keypoints.shape[0], 1))]
            )

            kp_rot = (rot_mat @ kp_homog.T).T

            keypoints = kp_rot

            factor = np.random.uniform(0.8, 1.2)

            image = np.clip(image * factor, 0, 255).astype(np.uint8)

            noise = np.random.normal(0, 5, image.shape)

            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, self.image_size - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, self.image_size - 1)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        heatmaps = np.zeros(
            (21, self.heatmap_size, self.heatmap_size),
            dtype=np.float32
        )

        sigma = 2

        for i, (x, y) in enumerate(keypoints):

            x_hm = x / self.image_size * self.heatmap_size
            y_hm = y / self.image_size * self.heatmap_size

            heatmaps[i] = np.exp(
                -((self.xx - x_hm) ** 2 +
                  (self.yy - y_hm) ** 2) / (2 * sigma ** 2)
            )

        return torch.tensor(image), torch.tensor(heatmaps)
from os.path import join

import torch
from numpy import random
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_folder, target_folder, transform=None, bbox_shift=40, eval=False, nflow=False, noise=False,
                 std=20, prob=0.2):
        self.image_folder = image_folder
        self.target_folder = target_folder
        self.transform = transform
        # 获取图像和地面真实值文件列表
        self.image_files = sorted(os.listdir(image_folder))
        self.target_files = sorted(os.listdir(target_folder))
        self.bbox_shift = bbox_shift
        self.needflow = nflow
        # noise parameter
        # guassian std
        self.noise = noise
        self.std = std
        # salt and pepper probability
        self.prob = prob
        self.eval = eval

    def __len__(self):
        return len(self.image_files)

    def add_noise(self, H, W, x_min, y_min, x_max, y_max):
        # 给bounding box添加扰动
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bounding_box = np.array([x_min, y_min, x_max, y_max])
        return bounding_box

    @staticmethod
    def add_image_noise(image, std, prob):
        # 添加高斯噪声
        # gauss = np.random.normal(0, std, image.shape)
        # noisy_image = image + gauss
        # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # # 添加椒盐噪声
        noisy_image = image
        salt_pepper = np.random.choice([0, 1, 2], size=image.shape, p=[1 - prob, prob / 2, prob / 2])
        noisy_image[salt_pepper == 1] = 0
        noisy_image[salt_pepper == 2] = 255

        return noisy_image

    @staticmethod
    def flowprocess(layer):
        # the nabla operator
        nablax = [[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]
        nablay = [[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]
        nabla = np.stack([nablax, nablay], axis=0)
        nabla = torch.FloatTensor(nabla).unsqueeze(0).permute(1, 0, 2, 3)
        padlayer = np.pad(layer, 1, mode='symmetric')

        layer_dis = distance_transform_edt(layer)
        layer_dis = torch.FloatTensor(layer_dis).unsqueeze(0).unsqueeze(0)

        layer_dis_out = distance_transform_edt(1 - layer)
        layer_dis_out = torch.FloatTensor(layer_dis_out).unsqueeze(0).unsqueeze(0)

        layer = torch.FloatTensor(layer)

        conv_layer_dis = F.conv2d(layer_dis, nabla, padding=0).squeeze(0)
        conv_layer_dis_out = F.conv2d(-layer_dis_out, nabla, padding=0).squeeze(0)

        conv_layer_dis = F.pad(conv_layer_dis, (1, 1, 1, 1), mode='constant', value=0)
        conv_layer_dis_out = F.pad(conv_layer_dis_out, (1, 1, 1, 1), mode='constant', value=0)

        flow_layer = conv_layer_dis * layer + conv_layer_dis_out * (1 - layer)

        norm_flow = F.normalize(flow_layer, dim=0, eps=1e-20)
        F_x, F_y = torch.split(norm_flow, 1, dim=0)
        norm_flow = torch.cat([-F_y, F_x], dim=0)
        return norm_flow

    def __getitem__(self, idx):
        # 读取图像和地面真实值
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        target_path = os.path.join(self.target_folder, self.target_files[idx])

        image = cv2.imread(image_path)
        if self.noise:
            image = self.add_image_noise(image, std=self.std, prob=self.prob)

        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        target = np.where(target > 0, 1, 0)
        nonzero_indices = np.nonzero(target)

        # 确定 bounding box 的坐标
        min_x, min_y = np.min(nonzero_indices[1]), np.min(nonzero_indices[0])
        max_x, max_y = np.max(nonzero_indices[1]), np.max(nonzero_indices[0])

        H, W = target.shape

        bounding_box = self.add_noise(H, W, min_x, min_y, max_x, max_y)
        # output:image:ndarray (H,W,3); mask:tensor(H,W);bounding_box:np (4)
        if self.eval:
            image_basename = os.path.basename(image_path)
            image_name = os.path.splitext(image_basename)[0]
            return image, torch.FloatTensor(target), bounding_box, image_name
        if self.needflow:
            maskflow = self.flowprocess(target)
            return image, torch.FloatTensor(target), bounding_box, maskflow
        else:
            return image, torch.FloatTensor(target), bounding_box




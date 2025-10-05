import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from MyDataset import MyDataset
from segment_anything import sam_model_registry
from typing import List, Tuple
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np
# from torchmetrics.functional import dice
from copy import deepcopy
from esp_module import *


class SAMESP(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            device,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            num_iterate = 100
    ):
        super(SAMESP, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.esp_module = ESPmodule(iterations=num_iterate)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.device = device
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    # image preprocess:
    @torch.no_grad()
    def preprocess(self, image: torch.tensor) -> torch.Tensor:
        """
        Expects a numpy with shape BxHxWxC
        """
        # 保持横纵比放缩至1024尺度
        target_size = self.get_preprocess_shape(image.shape[1], image.shape[2], self.image_encoder.img_size)
        image_scale =[]
        for i in range(image.shape[0]):
            # 获取单张图片scaling后的结果
            img_sample = np.array(resize(to_pil_image(image[i]), target_size))
            image_scale.append(img_sample)
        # image_scale = np.array(resize(to_pil_image(image), target_size))

        input_image_torch = torch.as_tensor(np.array(image_scale), device=self.device)
        # 由BxhxWxC# 转变成Bxcxhxw的tensor数据
        input_image_torch = input_image_torch.permute(0, 3, 1, 2).contiguous()
        self.image_scale_size = tuple(input_image_torch.shape[-2:])
        self.orgin_size = tuple(image.shape[1:3])
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad to 1xCx1024x1024

        h, w = input_image_torch.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(input_image_torch, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def preprocess_bbox(self, bbox: np.ndarray, original_size: Tuple[int, ...]):
        coords = bbox.reshape(-1, 2, 2)
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], 1024
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords.reshape(-1, 4)

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.size befor padding after scaling
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, image: np.ndarray, bbox: np.ndarray):
        """
        :param image: Expects a numpy array with shape BxHxWxC in uint8 format.
        :param bbox: box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
        :return:
        """

        input_image = self.preprocess(image)  # 得到放缩+padding之后的（B，3，1024，1024）
        image_embedding = self.image_encoder(input_image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        box = self.preprocess_bbox(bbox, (image.shape[1], image.shape[2]))
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=self.device)
            box_torch = box_torch[None, :]  # Tensor (B,1,4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        o = self.postprocess_masks(low_res_masks, self.image_scale_size, self.orgin_size)
        #（B, 1, H, W）
        masks = self.esp_module(o)
        #(B, H, W)
        return masks


import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg
import torch.nn.functional as F


def get_dice(SR, GT):
    intersection = ((SR == 1) & (GT == 1)).sum().item()
    dice = 2 * intersection / (SR.sum().item() + GT.sum().item() + 1e-10)
    return dice


def calculate_dice_batch(predictions, targets, threshold=0.5, epsilon=1e-8):
    # 将预测掩码通过 Sigmoid 函数转换为概率值
    predictions = torch.sigmoid(predictions)

    # 使用阈值将概率值转换为二进制预测
    binary_predictions = (predictions > threshold).float()

    batch_dice = []
    for i in range(predictions.size(0)):
        # 对每张图片计算 Dice 指标
        intersection = torch.sum(binary_predictions[i] * targets[i])
        union = torch.sum(binary_predictions[i]) + torch.sum(targets[i]) + epsilon
        dice = (2.0 * intersection + epsilon) / union
        batch_dice.append(dice.item())

    # 对 batch 中的每张图片的 Dice 指标求平均值
    average_dice = torch.mean(torch.tensor(batch_dice))

    return average_dice.item()


def calculate_iou(prediction, target):
    intersection = torch.logical_and(prediction, target).sum().float()
    union = torch.logical_or(prediction, target).sum().float()
    iou = (intersection + 1e-6) / (union + 1e-6)  # 加入一个小值以避免分母为0
    return iou.item()


# ---bd and bdsd--------
# distance_transform_edt只接受numpy类型的数据
def compute_df(img_gt):
    """
    compute the Euclidean distance map of binary mask
    input: segmentation, shape = (b, h, w)
    output: the Euclidean Distance Map (DM)

    """
    posmask = img_gt.astype(np.uint8)

    negmask = 1 - posmask
    # posdis是内部的点到边界的距离
    posdis = distance_transform_edt(posmask)
    # negdis是外部的点到边界的距离
    negdis = distance_transform_edt(negmask)
    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
    df = negdis + posdis
    df[boundary == 1] = 0

    return df


def calculate_bd(mask, gt):
    """
        mask:segmentation result of model,shape:(B,H,W）,type:tensor
        gt:groundtrue, shape:(B,H,W),type:tensor
        这两个都是二值图像
    """
    mask_numpy = mask.cpu().detach().numpy()
    gt_numpy = gt.cpu().detach().numpy()
    bdmap = compute_df(gt_numpy)
    # 计算mask的边界
    mask_boundary = skimage_seg.find_boundaries(mask_numpy, mode='inner').astype(np.uint8)
    mask_bd = mask_boundary * bdmap
    bd = mask_bd.sum() / (mask_boundary.sum() + 1e-10)
    # 下面计算bdsd:Boundary Distance Standard Deviation
    mask_bdu = (mask_bd - bd) * mask_boundary
    mask_bdsd = np.sum(mask_bdu ** 2) / (mask_boundary.sum() + 1e-10)
    bdsd = np.sqrt(mask_bdsd)
    return bd, bdsd


# def CFloss(predict, gtflow):
#     # predict: the output of the model
#     nabla = torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
#                           [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]])
#     u = torch.sigmoid(predict)
#     nabla = nabla.to(u.device, dtype=torch.float32)
#     nabla_u = F.conv2d(u.unsqueeze(1), weight=nabla, stride=1, padding=1)
#     cos_dis = F.cosine_similarity(nabla_u, gtflow, dim=1)
#     similarity = torch.mean(torch.abs(cos_dis))
#     return similarity

def CFloss(masks_pred, true_flows, n_classes=1):
    dx = [[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]
    dy = [[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]
    stack_nabla_d = np.stack([dx, dy] * n_classes, axis=0)
    masks_pred = masks_pred.unsqueeze(0)
    true_flows = true_flows.unsqueeze(0)
    u = torch.sigmoid(masks_pred)
    nabla = torch.FloatTensor(stack_nabla_d).unsqueeze(0).permute(1, 0, 2, 3)
    nabla = nabla.to(device=u.device, dtype=torch.float32)

    nabla_u = F.conv2d(F.pad(u, (1, 1, 1, 1), mode='constant', value=0), weight=nabla, groups=n_classes)

    similarity = 0.

    for u_layer, tflow_layer in zip(torch.split(nabla_u, 2, dim=1), torch.split(true_flows, 2, dim=1)):
        cos_dis = F.cosine_similarity(u_layer, tflow_layer, dim=1)
        similarity += torch.mean(F.relu(cos_dis) + F.relu(-cos_dis))

    return similarity / n_classes

if __name__ == "__main__":
    prediction = torch.tensor([[[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [0, 0, 1, 1],
                                [0, 0, 1, 1]]])

    target = torch.tensor([[[[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]],
                           [[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]]
                           ]])
    s = CFloss(prediction, target)
    print(s)

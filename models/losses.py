import torch
import torch.nn.functional as F
from torch import nn
import numpy

def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # torch.Size([8, 2, 256, 256])
    # torch.Size([8, 256, 256])
    print(target.size())
    target = target.long()  # 将数字或字符串转换为一个长整型
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    # input.squeeze(1)
    # print(input.shape, target.shape)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


# def BinaryDiceLoss(input, target, smooth = 1, p = 1, reduction = 'mean'):
#     assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
#
#     target = target.long()
#     input = input.sum(dim=1)
#
#     predict = input.contiguous().view(input.shape[0], -1)
#     target = target.contiguous().view(target.shape[0], -1)
#
#     # print('predict size:', predict.shape, 'target size:', target.shape)
#
#     num = 2. * torch.sum(torch.mul(predict, target), dim=1) + smooth
#     den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth
#
#     loss = 1 - num / den
#
#     if reduction == 'mean':
#         return loss.mean()
def DiceLoss(input, target, num_classes=2, smooth=1, p=1):
    target = target.long()

    target = target.squeeze(1)  # target：[batch_size, 1, w, h] -> [batch_size, w, h]
    target = torch.nn.functional.one_hot(target, num_classes)  # [batch_size, w, h, cls]
    target = target.permute(0, 3, 1, 2)  # [batch_size, w, h, cls] -> [batch_size, cls, w, h]

    # print("inputs.shape:", input)
    # print("targets.shape:", target.shape)

    # flatten label and prediction tensors
    input = input.contiguous().view(input.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = 2. * torch.sum(torch.mul(input, target), dim=1) + smooth
    den = torch.sum(input.pow(p) + target.pow(p), dim=1) + smooth

    loss = 1 - num / den
    # print('dice', loss.mean())
    return loss.mean()


def ce_dice(input, target, alpha=1):
    # print('input size:', input.shape, 'target size:', target.shape)
    # assert input.shape == target.shape, "predict & target shape do not match"
    # print('ce', cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255))
    return cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255) + alpha * DiceLoss(input, target)

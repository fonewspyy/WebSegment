import torch.nn.functional as F
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-10, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0] or predict.shape[1] == target.shape[1], "predict & target image size don't match"

        # NaN 및 Inf 검사
        if torch.isnan(predict).any() or torch.isnan(target).any():
            raise ValueError("Input contains NaN values")

        if torch.isinf(predict).any() or torch.isinf(target).any():
            raise ValueError("Input contains Inf values")

        intersection = torch.sum(target * predict)
        gt = torch.sum(target)
        pred = torch.sum(predict)
        total = gt + pred

        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        dice_loss = (1. - dice).mean()

        if torch.isnan(dice_loss):
            print(
                f'Warning: dice_loss is NaN. intersection: {intersection.item()}, gt: {gt.item()}, pred: {pred.item()}')
            dice_loss = torch.tensor(1.0, device=dice_loss.device)

        return dice_loss


class MultiSoftDiceLoss(nn.Module):
    def __init__(self):
        super(MultiSoftDiceLoss, self).__init__()
        self.dice = BinaryDiceLoss()

    def forward(self, predict, target, organ_number):
        # predict를 sigmoid를 거쳐 확률 값으로 변환

        predict = F.sigmoid(predict)

        total_loss = []
        # batch size
        B = predict.shape[0]

        for b in range(B):
            # loss 연산을 수행할 organ class
            # ignore background
            for organ in range(organ_number):
                dice_loss = self.dice(predict[b, organ], target[b, organ])
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]

class Multi_BCELoss(nn.Module):
    def __init__(self):
        super(Multi_BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, organ_num):
        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # ignore background
            for organ in range(organ_num):
                bce_loss = self.criterion(predict[b, organ], target[b, organ])
                total_loss.append(bce_loss)

        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]

import torch.nn.functional as F
import torch
import numpy as np

def DSC_IoU_EachClass(predicted, target, out_classes=2, smooth=1e-10):
    n_class = out_classes

    predicted = predicted.squeeze(0)
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()
    predicted = torch.argmax(predicted, dim=0)

    target = target.squeeze(0)
    target = torch.argmax(target, dim=0)

    dice = torch.ones(n_class).float()
    iou = torch.ones(n_class).float()

    for i in range(n_class):
        # predicted에서 i번째 클래스라고 예측한 위치
        predicted_temp = torch.eq(predicted, i)
        # 실제로 i번째 클래스인 위치
        target_temp = torch.eq(target, i)
        # i번째 클래스를 올바르게 맞춘 개수
        intersection = predicted_temp & target_temp
        intersection = intersection.float().sum()
        total = target_temp.float().sum() + predicted_temp.float().sum()
        union = total-intersection

        dice[i] = (2*intersection+smooth) / (total+smooth)
        iou[i] = (intersection + smooth) / (union + smooth)

    return dice, iou
from models.UNet_Partial import UNet_Partial
from losses.MultiSoftDiceLoss import MultiSoftDiceLoss, Multi_BCELoss
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import numpy as np

# 가중치 초기화 함수
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_weights(model, optimizer, lr_scheduler, scaler, exp):
    saved_model_path = f'SavedModel/best_{exp}.pt'
    print(f"====> setting weight: {saved_model_path}")
    saved_model = torch.load(saved_model_path)

    model.load_state_dict(saved_model['model'])
    optimizer.load_state_dict(saved_model['optimizer'])
    lr_scheduler.load_state_dict(saved_model['lr_scheduler'])
    scaler.load_state_dict(saved_model['scaler'])

    return model, optimizer, lr_scheduler, scaler


def continual_segmentation(train_loader, valid_loader, epoch, lr, out_class, exp, device, text_embeddings,
                           pretrain=None, controller_map=None):
    model = UNet_Partial(in_channels=1, out_channels=out_class, text_embeddings=text_embeddings).to(device)
    model.apply(weights_init)

    bce_criterion = Multi_BCELoss().to(device)
    dice_criterion = MultiSoftDiceLoss().to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    if pretrain is not None:
        model = set_model(pretrain, model, controller_map).to(device)

    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-6)
    # optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-6)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=lr * 0.01)

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists(f'./SavedModel/best_{exp}.pt'):
        model, optimizer, scheduler, scaler = set_weights(model, optimizer, scheduler, scaler, exp)

    # for tensorboard
    writer = SummaryWriter()
    # empty cache for preventing cuda out of memory issue
    torch.cuda.empty_cache()
    gc.collect()

    # train model
    train_logs_list, valid_logs_list = [], []
    best_valid_loss = np.inf

    save_check = 0
    for i in range(epoch):
        # 20 epoch 이상 best valid loss가 갱신되지 않으면 early stop (prevent overfitting)
        if save_check > 20:
            print('Early stopping at epoch {}'.format(i))
            break

        print(f'EPOCH {i + 1}/{epoch}')

        train_loss = train_fn(train_loader, model, optimizer, device, bce_criterion, dice_criterion, scaler)
        valid_loss = eval_fn(valid_loader, model, device, bce_criterion, dice_criterion)

        train_logs_list.append(train_loss)
        valid_logs_list.append(valid_loss)

        if valid_loss < best_valid_loss:
            save_check = 0
            best_valid_loss = valid_loss
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        }, f'./SavedModel/best_{exp}.pt')
            print('Model Saved')
        else:
            save_check += 1

        # scheduler.step(valid_loss)
        scheduler.step()

        # tensorboard
        writer.add_scalar(f"{exp}/Loss/train", train_loss, i)
        writer.add_scalar(f"{exp}/Loss/valid", valid_loss, i)

        # if pretrain is None:
        print(
            f'Epoch {i + 1} finished : train loss : {train_loss}, valid loss : {valid_loss}, lr :{scheduler.get_last_lr()[0]}')

    writer.flush()
    writer.close()


def train_fn(loader, model, optimizer, device, bce_criterion, dice_criterion, scaler):
    model.train()
    total_loss = 0.0

    with tqdm(loader) as pbar:
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type='cuda'):
                pred = model(images)

                # 비정상적으로 큰 값 또는 NaN/Inf 값 확인
                if torch.isnan(pred).any() or torch.isinf(labels).any():
                    raise ValueError("NaN or Inf values detected in model output")

                bce_loss = bce_criterion(pred, labels, labels.shape[1])
                dice_loss = dice_criterion(pred, labels, labels.shape[1])
                loss = bce_loss + dice_loss
            total_loss += loss.item()

            # loss.backward()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"Train Loss": loss.item()})

            # GPU에서 사용하지 않는 메모리를 반환
            torch.cuda.empty_cache()
    return total_loss / len(loader)


def eval_fn(loader, model, device, bce_criterion, dice_criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        with tqdm(loader) as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(images)
                    bce_loss = bce_criterion(pred, labels, labels.shape[1])
                    dice_loss = dice_criterion(pred, labels, labels.shape[1])
                    loss = bce_loss + dice_loss
                total_loss += loss.item()

                pbar.set_postfix({"Valid Loss": loss.item()})

    return total_loss / len(loader)


def set_model(saved_model, new_model, controller_map=None):
    saved_model_path = f'SavedModel/best_{saved_model}.pt'
    print(f"====> loading pretrained: {saved_model_path}")
    old_model_state = torch.load(saved_model_path)['model']

    print(f'except state (bg, adding class) : {new_model.state_dict().keys() - old_model_state.keys()}')

    new_model.load_state_dict(old_model_state, strict=False)

    # controller_map = {1: 4}
    if controller_map is not None:
        for old_idx, new_idx in controller_map.items():
            print(f"Copying weights from controller {old_idx} to controller {new_idx}")

            # 기존 controller의 가중치
            old_controller_state = {k: v for k, v in old_model_state.items() if f'controllers.{old_idx}' in k}
            # controller.1. => controller.4로 변경해서 복사
            new_controller_state = {k.replace(f'controllers.{old_idx}', f'controllers.{new_idx}'): v
                                    for k, v in old_controller_state.items()}
            # new_controller_state만 new_model에 transfer
            new_model.load_state_dict(new_controller_state, strict=False)

    return new_model
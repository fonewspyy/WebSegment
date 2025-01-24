import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from Dataloader.ContinualDataset import ContinualSegmentationDataset
from continual.continual_inference import mmwhs_continual_inference
from continual.continual_segmentation import continual_segmentation

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
MMWHS_DIR = 'dataset/ct_train'
device = 'cuda:0'

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Print the name of the GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

# Load data
train_image260340 = np.load(f'{MMWHS_DIR}/npz/train_image260340.npz')['data'].astype(np.float32)/255.0
train_label = np.load(f'{MMWHS_DIR}/npz/train_label.npz')['data'].astype(np.float32)

valid_image260340 = np.load(f'{MMWHS_DIR}/npz/valid_image260340.npz')['data'].astype(np.float32)/255.0
valid_label = np.load(f'{MMWHS_DIR}/npz/valid_label.npz')['data'].astype(np.float32)

# # baseline
# train_set = SegmentationDataset(train_image260340, train_label, outclass=8)
# valid_set = SegmentationDataset(valid_image260340, valid_label, outclass=8)
#
# train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
# valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
#
# # lr cosine 100 / 1e-4 시작
# Segmentation(train_loader, valid_loader, device, 100, "cardiac_unet_baseline", 1e-4, "unet")
# inference(valid_set, valid_loader, "cardiac_unet_baseline", device, "unet", out_class=8)

# Continual
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class_name = {
    1: 'a left ventricle blood cavity', # LVC
    2: 'a right ventricle blood cavity', # RVC
    3: 'a left atrium blood cavity', #LAC
    4: 'a right atrium blood cavity', # RAC
    5: 'a myocardium of the left ventricle', #MYO
    6: 'an ascending aorta', # AA
    7: 'a pulmonary artery', # PA
}

# Model Saved
# EPOCH 13 : train loss : 0.4617697460789269, valid loss : 0.5105077009272924, lr :8.438508174347009e-05
# group_2 정규화 위해서 학습 중단

# 뒤 요소를 분리
groups = [[1, 5], [5, 2], [7, 2], [4, 2]]

for group_idx in range(len(groups)):
    group = groups[group_idx]
    print(f"Processing group {group_idx + 1}/{len(groups)}: {group}")

    # Generate text embeddings
    mmwhs_class_text_gr = ["a computerized tomography of a background"]
    template = "a computerized tomography of "
    grouped_classes = [class_name[cls] for cls in group]
    grouped_sentence = f"{template}{' and '.join(grouped_classes)}"
    mmwhs_class_text_gr.append(grouped_sentence)

    class_order = ['BG', ' + '.join(grouped_classes)]
    print(f"  Grouped classes: {grouped_classes}")
    print(f"  Class order: {class_order}")

    not_grouped_cls = [class_name[cls] for cls in range(1, 8) if cls not in group]
    for cls_name in not_grouped_cls:
        mmwhs_class_text_gr.append(f"{template}{cls_name}")
        class_order.append(cls_name)

    # CLIP
    text_tokens = clip.tokenize(mmwhs_class_text_gr).to(device)
    print("  Tokenizing text embeddings...")

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
    print("  Text embeddings generated.")

    # Label mapping
    label_mapping = {0: 0}
    cur_cls_id = 2
    for class_id in range(1, 8):
        if class_id in group:
            label_mapping[class_id] = 1
        else:
            label_mapping[class_id] = cur_cls_id
            cur_cls_id += 1
    print(f"  Label mapping: {label_mapping}")

    target_class = list(range(1, 8))

    # Dataset creation
    train_set = ContinualSegmentationDataset(
        train_image260340, train_label,
        target_class=target_class,
        out_class=7, label_mapping=label_mapping
    )
    valid_set = ContinualSegmentationDataset(
        valid_image260340, valid_label,
        target_class=target_class,
        out_class=7, label_mapping=label_mapping
    )
    print("  Datasets created.")

    # DataLoader creation
    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False
    )
    print("  DataLoaders created.")

    #Training and inference
    # print(f"  Starting continual segmentation for group {group_idx + 1}...")
    # continual_segmentation(
    #     train_loader, valid_loader, 100, 1e-5, 7, f"mmwhs_unet_group_{group_idx}_50_norm",
    #     text_embeddings=text_embeddings, pretrain=None, device=device
    # )
    print("  Continual segmentation complete.")
    mmwhs_continual_inference(
        valid_set, valid_loader, f"mmwhs_unet_group_{group_idx}_50_norm",
        device, 7, text_embeddings, target_class=[1], class_order=class_order
    )

    # Separate classes for the next step
    sep_cls = group[1]
    not_sep_cls = group[0]
    new_class_text = mmwhs_class_text_gr.copy()
    new_class_text[1] = template + class_name[not_sep_cls]
    new_class_text.append(f"{template}{class_name[sep_cls]}")
    new_class_order = class_order.copy()
    new_class_order[1] = class_name[not_sep_cls]
    new_class_order.append(class_name[sep_cls])
    print(f"  Updated class order for separation: {new_class_order}")

    text_tokens = clip.tokenize(new_class_text).to(device)
    print("  Tokenizing updated text embeddings...")

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
    print("  Updated text embeddings generated.")

    new_label_mapping = label_mapping.copy()
    new_label_mapping[sep_cls] = 7
    print(f"  New label mapping: {new_label_mapping}")

    train_set = ContinualSegmentationDataset(
        train_image260340, train_label,
        target_class=target_class, out_class=8,
        label_mapping=new_label_mapping
    )
    valid_set = ContinualSegmentationDataset(
        valid_image260340, valid_label,
        target_class=target_class, out_class=8,
        label_mapping=new_label_mapping
    )
    print("  New datasets created.")

    train_loader = DataLoader(
        train_set, batch_size=8,
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1, shuffle=False
    )
    print("  New DataLoaders created.")

    # print(f"  Starting continual segmentation for separated class {sep_cls}...")
    # continual_segmentation(
    #     train_loader, valid_loader, 100, 1e-5, 8,
    #     f"mmwhs_unet_group_{group_idx}_sep{sep_cls}_50_norm",
    #     text_embeddings=text_embeddings, pretrain=f"mmwhs_unet_group_{group_idx}_50_norm", device=device
    # )
    print("  Continual segmentation complete for separated class.")
    mmwhs_continual_inference(
        valid_set, valid_loader,
        f"mmwhs_unet_group_{group_idx}_sep{sep_cls}_50_norm",
        device, 8, text_embeddings, target_class=[1, 7], class_order=new_class_order
    )

del train_image260340
del train_label
del valid_image260340
del valid_label
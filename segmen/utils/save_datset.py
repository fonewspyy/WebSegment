import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def window_CT(slice, min=-260, max=340):
  # kidney HU = +20 to +45
  # best visualize = -260 to +340
  sub=abs(min)
  diff=abs(min-max)

  img=slice+sub
  img[img<=0]=0   # min normalization
  img[img>diff]=diff  # max normalization
  img=img/diff
  # img = np.clip(slice, min, max)
  # img = (img-min)/(max-min)
  return img


def save_cardiac_dataset():
    DATA_DIR = 'dataset/ct_train'

    cardiac_mapping = {
        500: 1,
        600: 2,
        420: 3,
        550: 4,
        205: 5,
        820: 6,
        850: 7,
        0: 0
    }

    # df로 저장
    ORIGINAL_IMAGES = []
    MASK_IMAGES = []

    CASES = sorted(os.listdir(DATA_DIR))
    for c in CASES:
        case = c.split('.')[0]
        if case.endswith('image'):
            ORIGINAL_IMAGES.append(os.path.join(DATA_DIR + case + ".nii.gz"))
        elif case.endswith('label'):
            MASK_IMAGES.append(os.path.join(DATA_DIR + case + ".nii.gz"))

    df_data = pd.DataFrame({'image': ORIGINAL_IMAGES, 'label': MASK_IMAGES})
    print(f"Number of Data : {len(df_data)}")

    # split train set / valid set
    train_df, valid_df = train_test_split(df_data, test_size=0.2, random_state=42)
    print(f"number of train patient : {len(train_df)}, number of valid patient : {len(valid_df)}")

    data_image260340_npz_array = []
    data_label_npz_array = []

    for idx in range(len(train_df)):
        obj_img = nib.load(train_df['image'].iloc[idx])
        obj_lbl = nib.load(train_df['label'].iloc[idx])

        img = obj_img.get_fdata(dtype=np.float32)
        lbl = obj_lbl.get_fdata(dtype=np.float32)

        # class mapping
        # 단, mapping에 존재하지 않는 경우는 NONE으로 처리함 주의
        lbl = np.vectorize(cardiac_mapping.get)(lbl)

        # [d, h, w]로 변환
        img = np.rot90(np.transpose(img, (2, 0, 1)), k=1, axes=(1, 2))
        lbl = np.rot90(np.transpose(lbl, (2, 0, 1)), k=1, axes=(1, 2))

        # window CT
        image260340 = np.asarray([window_CT(_, min=-260, max=340) for _ in img])

        image260340 = (image260340 * 255).astype(np.uint8)
        label = lbl.astype(np.uint8)

        data_image260340_npz_array.append(image260340)
        data_label_npz_array.append(label)

        if idx % 5 == 0:
            print(f"saved {idx}th train image")

    concat_image260340 = np.concatenate(data_image260340_npz_array, axis=0)
    concat_label = np.concatenate(data_label_npz_array, axis=0)

    np.savez_compressed(f"{DATA_DIR}/npz/train_image260340.npz", data=concat_image260340)
    np.savez_compressed(f"{DATA_DIR}/npz/train_label.npz", data=concat_label)

    data_image260340_npz_array = []
    data_label_npz_array = []
    for idx in range(len(valid_df)):
        obj_img = nib.load(valid_df['image'].iloc[idx])
        obj_lbl = nib.load(valid_df['label'].iloc[idx])

        img = obj_img.get_fdata(dtype=np.float32)
        lbl = obj_lbl.get_fdata(dtype=np.float32)

        lbl = np.vectorize(cardiac_mapping.get)(lbl)

        # [d, h, w]로 변환
        img = np.rot90(np.transpose(img, (2, 0, 1)), k=1, axes=(1, 2))
        lbl = np.rot90(np.transpose(lbl, (2, 0, 1)), k=1, axes=(1, 2))

        # window CT
        image260340 = np.asarray([window_CT(_, min=-260, max=340) for _ in img])

        image260340 = (image260340 * 255).astype(np.uint8)
        label = lbl.astype(np.uint8)

        data_image260340_npz_array.append(image260340)
        data_label_npz_array.append(label)

        if idx % 5 == 0:
            print(f"saved {idx}th train image")

    concat_image260340 = np.concatenate(data_image260340_npz_array, axis=0)
    concat_label = np.concatenate(data_label_npz_array, axis=0)

    np.savez_compressed(f"{DATA_DIR}/npz/valid_image260340.npz", data=concat_image260340)
    np.savez_compressed(f"{DATA_DIR}/npz/valid_label.npz", data=concat_label)
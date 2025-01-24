import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from models.UNet_Partial import UNet_Partial
from utils.postprocess import colour_code_segmentation
from matrices.calculated_matrices import DSC_IoU_EachClass

def mmwhs_continual_inference(valid_set, valid_loader, exp_num, device, out_class, text_embedding, target_class, class_order):
    print('-' * 10 + f" Starting Inference: {exp_num} " + '-' * 10)

    # Load the best model
    print("Loading the best model...")
    best_checkpoint = torch.load(f'./SavedModel/best_{exp_num}.pt')
    best_model = UNet_Partial(in_channels=1, out_channels=out_class, text_embeddings=text_embedding).to(device)
    best_model.load_state_dict(best_checkpoint['model'])
    print("Model loaded successfully.")

    best_model.eval()

    # Predict random samples
    print("Generating random predictions for visualization...")
    with torch.no_grad():
        primg_num = 0
        for i in range(len(valid_set)):
            if primg_num > 10:
                break
            idx = np.random.randint(0, len(valid_set))
            image, label = valid_set[idx]

            label = np.argmax(label, axis=0)
            if set(target_class).issubset(label.flatten().tolist()):
                primg_num += 1

                print(f"Processing image {primg_num} with random index {idx}...")

                x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

                # Predict test image
                pred_mask = best_model(x_tensor)
                probs = torch.sigmoid(pred_mask)
                pred_mask = probs.detach().squeeze(0).cpu().numpy()
                pred_mask = np.transpose(pred_mask, (1, 2, 0))  # CHW -> HWC

                ch_num = image.shape[0]
                plt.figure(figsize=(5 * ch_num + 10, 5))

                HU = [[-260, 340], [-190, -30], [-29, 150]]
                for j in range(ch_num):
                    plt.subplot(1, ch_num + 2, j + 1)
                    plt.title(f'Input image {HU[j]}')
                    plt.imshow(image[j], cmap='gray')

                plt.subplot(1, ch_num + 2, ch_num + 1)
                plt.title('Ground-truth')
                plt.imshow(colour_code_segmentation(label))

                plt.subplot(1, ch_num + 2, ch_num + 2)
                plt.title("Prediction")
                pred_mask = (pred_mask > 0.5).astype(float)
                plt.imshow(colour_code_segmentation(np.argmax(pred_mask, axis=2)))

                if not os.path.exists(f'result/exp_{exp_num}'):
                    os.mkdir(f'result/exp_{exp_num}')
                save_path = f'result/exp_{exp_num}/prediction_{i}.png'
                plt.savefig(save_path)
                print(f"Prediction saved to {save_path}")
                plt.show()

    # Calculate and log metrics for each class
    print("Calculating metrics for each class...")
    predictions = []
    for batch_idx, (images, labels) in enumerate(valid_loader):
        #print(f"Processing batch {batch_idx + 1}/{len(valid_loader)}...")
        images = images.to(device)
        labels = labels.to(device)

        output = best_model(images)
        dice, iou = DSC_IoU_EachClass(output, labels, out_class)

        prediction = {}
        for idx in range(out_class):
            prediction[f'{class_order[idx].lower()}_dice'] = dice[idx].item()
            prediction[f'{class_order[idx].lower()}_iou'] = iou[idx].item()

        predictions.append(prediction)

    df_csv = pd.DataFrame(predictions)
    result_dir = f'result/exp_{exp_num}/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    csv_path = f"{result_dir}prediction_{exp_num}.csv"
    df_csv.to_csv(csv_path)
    print(f"Metrics saved to {csv_path}")

    # Print mean metrics
    print("Calculating mean metrics for each class...")
    for c in range(out_class):
        mean_dice = df_csv[f'{class_order[c].lower()}_dice'].mean()
        mean_iou = df_csv[f'{class_order[c].lower()}_iou'].mean()
        print(f'{class_order[c]} DSC : {mean_dice:.4f}')
        print(f'{class_order[c]} IoU : {mean_iou:.4f}')

    print('-' * 10 + f" Inference Completed: {exp_num} " + '-' * 10)

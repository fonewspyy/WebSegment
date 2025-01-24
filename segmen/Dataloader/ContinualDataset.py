from torch.utils.data import Dataset
import numpy as np
from utils.preprocess import onehot_encoding

class ContinualSegmentationDataset(Dataset):
    def __init__(
            self,
            images260340,
            labels,
            target_class,
            out_class,
            label_mapping
    ):
        self.images260340 = images260340
        self.labels = labels
        self.target_class = target_class
        self.out_class = out_class
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        image260340 = self.images260340[index]
        label = self.labels[index]

        new_label = np.where(np.isin(label, list(self.target_class)),
                    np.vectorize(self.label_mapping.get)(label, 0), 0)

        image = image260340[np.newaxis, ...]

        # onehot encoding with dynamic number of classes
        new_label = onehot_encoding(new_label, self.out_class)

        return image, new_label

    def __len__(self):
        return len(self.labels)
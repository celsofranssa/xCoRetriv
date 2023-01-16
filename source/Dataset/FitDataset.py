import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FitDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_path, text_tokenizer, label_tokenizer, text_max_length,
                 labels_max_length):
        super(FitDataset, self).__init__()
        self.samples = []
        self._reshape_samples(samples, pseudo_labels)
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.text_max_length = text_max_length
        self.labels_max_length = labels_max_length

        self._load_ids(ids_path)

    def _reshape_samples(self, samples, pseudo_labels):
        # text_sizes = []
        # label_sizes = []

        for sample in tqdm(samples, desc="Reshaping samples"):
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                # a = {
                #     "text_idx": sample["text_idx"],
                #     "text": " ".join([w[0] for w in sample["keywords"]]),
                #     "label_idx": label_idx,
                #     "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
                # }
                self.samples.append({
                    "text_idx": sample["text_idx"],
                    "text": " ".join([w[0] for w in sample["keywords"]]),
                    "label_idx": label_idx,
                    "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
                })
                #
                # text_sizes.append(len(a["text"].split()))
                # label_sizes.append(len(a["label"].split()))

        # print(f"Texts: {np.quantile(text_sizes, [0.5, 0.75,0.9])}")
        # print(f"Labels: {np.quantile(label_sizes, [0.5, 0.75, 0.9])}")


    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):

        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.text_tokenizer.encode(text=sample["text"], max_length=self.text_max_length,
                                           padding="max_length", truncation=True, add_special_tokens=False)
            ),
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.label_tokenizer.encode(text=sample["label"], max_length=self.labels_max_length,
                                            padding="max_length", truncation=True, add_special_tokens=False)
            ),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )

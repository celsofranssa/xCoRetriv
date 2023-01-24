import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RetrieverFitDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_path, text_tokenizer, label_tokenizer, text_max_length,
                 labels_max_length):
        super(RetrieverFitDataset, self).__init__()
        self.samples = []
        self.ids = self._load_ids(ids_path)
        self._reshape_samples(samples, pseudo_labels)
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.text_max_length = text_max_length
        self.labels_max_length = labels_max_length

    def _reshape_samples(self, samples, pseudo_labels):
        for idx in tqdm(self.ids, desc="Reshaping samples"):
            sample = samples[idx]
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                self.samples.append({
                    "text_idx": sample["text_idx"],
                    "text": sample["text"],
                    "label_idx": label_idx,
                    "label": label + " " + " ".join(x[0] for x in pseudo_labels[label_idx]),
                })

    def _get_pseudo_labels(self, pseudo_labels):
        return random.choices(
            [label for (label, _) in pseudo_labels],
            [weight for (_, weight) in pseudo_labels],
            k=8
        )

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, sample):

        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.text_tokenizer.encode(text=sample["text"], max_length=self.text_max_length,
                                           padding="max_length", truncation=True)
            ),
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.label_tokenizer.encode(text=sample["label"], max_length=self.labels_max_length,
                                            padding="max_length", truncation=True)
            ),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )

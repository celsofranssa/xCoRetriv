import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ReRankerPredictDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, rankings, pseudo_labels, ids_path, text_tokenizer, label_tokenizer, text_max_length,
                 labels_max_length):
        super(ReRankerPredictDataset, self).__init__()
        self.samples = []
        self.pseudo_labels = pseudo_labels
        self._load_ids(ids_path)
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.text_max_length = text_max_length
        self.labels_max_length = labels_max_length
        texts = {}
        labels = {}

        for sample in tqdm(samples, desc="Reading samples"):
            texts[sample["text_idx"]] = sample["text"]
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels[label_idx] = label

        # for idx in tqdm(self.ids, desc="Reading samples"):
        #     sample = samples[idx]
        #     texts[sample["text_idx"]] = sample["text"]
        #     for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
        #         labels[label_idx] = label

        for text_idx, labels_scores in tqdm(rankings["tail"].items(), desc="Reading ranking"):
            text_idx = int(text_idx.split("_")[-1])
            for label_idx, score in labels_scores.items():
                label_idx = int(label_idx.split("_")[-1])
                a = {
                    "text_idx": text_idx,
                    "text": texts[text_idx],
                    "label_idx": label_idx,
                    "label": labels[label_idx] + " " + " ".join(x[0] for x in pseudo_labels[label_idx]),
                    "retriever_score": score
                }
                #print(a)
                self.samples.append(a)

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
            "retriever_score": sample["retriever_score"]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )

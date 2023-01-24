import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RetrieverLabelDataset(Dataset):

    def __init__(self, samples, pseudo_labels, ids_path, label_tokenizer, label_max_length):
        super(RetrieverLabelDataset, self).__init__()
        self.samples = []
        self.ids = self._load_ids(ids_path)
        self._reshape_samples(samples, pseudo_labels)
        self.label_tokenizer = label_tokenizer
        self.label_max_length = label_max_length

    def _reshape_samples(self, samples, pseudo_labels):
        labels = {}
        for idx in tqdm(self.ids, desc="Reshaping samples"):
            for label_idx, label in zip(samples[idx]["labels_ids"], samples[idx]["labels"]):
                labels[label_idx] = label

        for label_idx, label in labels.items():
            self.samples.append({
                "label_idx": label_idx,
                "label": label + " " + " ".join(x[0] for x in pseudo_labels[label_idx])
            })

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, sample):
        return {

            "label_idx": sample["label_idx"],
            "text": torch.tensor(
                self.label_tokenizer.encode(text=sample["text"], max_length=self.label_max_length,
                                           padding="max_length", truncation=True)
            )
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )

import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RetrieverTextDataset(Dataset):

    def __init__(self, samples, ids_path, text_tokenizer, text_max_length):
        super(RetrieverTextDataset, self).__init__()
        self.samples = []
        self.ids = self._load_ids(ids_path)
        self._reshape_samples(samples)
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length

    def _reshape_samples(self, samples):
        texts = {}
        for idx in tqdm(self.ids, desc="Reshaping samples"):
            texts[samples[idx]["text_idx"]] = samples[idx]["text_idx"]

        for text_idx, text in texts.items():
            self.samples.append({
                "text_idx": text_idx,
                "text": text
            })

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, sample):
        return {

            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.text_tokenizer.encode(text=sample["text"], max_length=self.text_max_length,
                                           padding="max_length", truncation=True)
            )
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )

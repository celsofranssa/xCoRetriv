import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.FitDataset import FitDataset


class XMTCDataModule(pl.LightningDataModule):

    def __init__(self, params, text_tokenizer, label_tokenizer, ranking, fold_idx):
        super(XMTCDataModule, self).__init__()
        self.params = params
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.ranking = ranking
        self.fold_idx = fold_idx

    def prepare_data(self):
        with open(self.params.dir + f"samples_with_keywords.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

        with open(f"{self.params.dir}fold_{self.fold_idx}/pseudo_labels.pkl", "rb") as pseudo_labels_file:
            self.pseudo_labels = pickle.load(pseudo_labels_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = FitDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/train.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.label_max_length
            )

            self.val_dataset = FitDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/val.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage == "predict":
            self.predict_dataset = FitDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/test.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.label_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

import pickle
from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.DataModule.RetrieverDataModule import RetrieverDataModule
from source.DataModule.XMTCDataModule import XMTCDataModule
from source.callback.PredictionWriter import PredictionWriter
from source.callback.RetrieverPredictionWriter import RetrieverPredictionWriter
from source.model.ReRankerModel import ReRankerXMTCModel
from source.model.RetrieverModel import RetrieverModel


class PredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            # data
            dm = RetrieverDataModule(
                params=self.params.data,
                text_tokenizer=self.get_tokenizer(self.params.model.tokenizer),
                label_tokenizer=self.get_tokenizer(self.params.model.tokenizer),
                fold_idx=fold_idx)

            # model
            model = RetrieverModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
            )

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[RetrieverPredictionWriter(self.params.prediction)]
            )

            # predicting
            dm.prepare_data()
            dm.setup("predict")

            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=dm,

            )

    def get_tokenizer(self, params):
        return AutoTokenizer.from_pretrained(
            params.architecture
        )

    def _get_rankings(self):
        with open(f"{self.params.ranking.dir}{self.params.ranking.name}.rnk", "rb") as ranking_file:
            return pickle.load(ranking_file)

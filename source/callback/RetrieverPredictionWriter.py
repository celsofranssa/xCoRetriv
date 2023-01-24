from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class RetrieverPredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(RetrieverPredictionWriter, self).__init__(params.write_interval)
        self.params = params
        self.checkpoint_dir = f"{self.params.dir}fold_{self.params.fold_idx}/"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        predictions = []

        # return {
        #     "text_idx": batch["text_idx"],
        #     "label_idx": batch["label_idx"],
        #     "retriever_score": batch["retriever_score"],
        #     "reranker_score": self(batch["text"], batch["label"])
        # }

        for text_idx, label_idx, retriever_score, reranker_score in zip(
                prediction["text_idx"].tolist(),
                prediction["label_idx"].tolist(),
                prediction["retriever_score"].tolist(),
                prediction["reranker_score"].tolist(),
        ):
            predictions.append({
                "text_idx": text_idx,
                "label_idx": label_idx,
                "retriever_score": retriever_score,
                "reranker_score": reranker_score,
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        torch.save(
            predictions,
            f"{self.checkpoint_dir}{dataloader_idx}_{batch_idx}.prd"
        )

import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from source.metric.RetrieverMRRMetric import RetrieverMRRMetric
from source.pooling.NoPooling import NoPooling


class RetrieverModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):
        super(RetrieverModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        # pooling
        self.pool = NoPooling()

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = RetrieverMRRMetric(hparams.metric)

    def forward(self, text, label):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_idx, text, label_idx, label = batch["text_idx"], batch["text"], batch["label_idx"], batch["label"]
        text_rpr, label_rpr = self.encoder(text), self.encoder(label)
        train_loss = self.loss(text_idx, text_rpr, label_idx, label_rpr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text_idx, text, label_idx, label = batch["text_idx"], batch["text"], batch["label_idx"], batch["label"]
        text_rpr, label_rpr = self.encoder(text), self.encoder(label)
        self.mrr.update(text_idx, text_rpr, label_idx, label_rpr)

    def validation_epoch_end(self, outs):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            text_idx, text, = batch["text_idx"], batch["text"]
            text_rpr = self.pool(self.encoder(text))

            return {
                "text_idx": text_idx,
                "text_rpr": text_rpr,
                "modality": "text"
            }
        else:
            label_idx, label = batch["label_idx"], batch["label"]
            label_rpr = self.pool(self.encoder(label))

            return {
                "label_idx": label_idx,
                "label_rpr": label_rpr,
                "modality": "label"
            }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                      cycle_momentum=False)
        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )

import torch
from pytorch_lightning import LightningModule
from transformers import BertModel

class BertEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )

    def forward(self, sample, sample_att):
        return self.encoder(
            sample, sample_att
        )


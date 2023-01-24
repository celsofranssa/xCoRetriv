import torch
from pytorch_lightning import LightningModule
from transformers import BertModel


class RetrieverBertEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, pooling):
        super(RetrieverBertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        self.pooling = pooling

    def forward(self, sample):
        attention_mask = torch.where(sample > 0, 1, 0)
        encoder_outputs = self.encoder(sample, attention_mask)
        return self.pooling(encoder_outputs, attention_mask)

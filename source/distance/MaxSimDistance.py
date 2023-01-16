import torch
from pytorch_metric_learning.distances import BaseDistance


class MaxSimDistance(BaseDistance):

    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, text_rpr, label_rpr):
        m = torch.einsum('b i j, c k j -> b c i k', text_rpr, label_rpr)
        m = torch.max(m, -1).values.sum(dim=-1)
        return torch.nn.functional.normalize(m, p=2, dim=-1)
        # print(f"m ({m.shape}):\n{m}\n")
        # return m

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

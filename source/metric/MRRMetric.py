import pickle
import nmslib
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__(compute_on_cpu=True)
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.ranking = {}

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _get_scores(self, text_rpr, label_rpr):
        m = torch.einsum('b i j, c k j -> b c i k', text_rpr, label_rpr)
        m = torch.max(m, -1).values.sum(dim=-1)
        return torch.nn.functional.normalize(m, p=2, dim=-1)

    def update(self, texts_ids, texts_rpr, labels_ids, labels_rpr):
        #
        # scores = torch.einsum('b i j, c k j -> b c i k', texts_rpr, labels_rpr)
        # scores = torch.max(scores, -1).values.sum(dim=-1)
        # scores = torch.nn.functional.normalize(scores, p=2, dim=-1)
        scores = self._get_scores(texts_rpr, labels_rpr)

        # print(f"label_idx({scores.shape}):\n{scores}\n")
        # print(f"label_idx({scores[0][0].shape}):\n{scores[0][0]}\n")

        for i, text_idx in enumerate(texts_ids.tolist()):
            for j, label_idx in enumerate(labels_ids.tolist()):
                if f"text_{text_idx}" not in self.ranking:
                    self.ranking[f"text_{text_idx}"] = {}
                self.ranking[f"text_{text_idx}"][f"label_{label_idx}"] = scores[i][j].item()

    def compute(self):

        # eval
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in self.ranking.keys()}),
            Run(self.ranking),
            ["mrr"]
        )
        return m

    def reset(self) -> None:
        self.ranking = {}

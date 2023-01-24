import hydra
import os
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.RetrieverFitHelper import RetrieverFitHelper


def fit(params):
    if params.model.type == "Retriever":
        helper = RetrieverFitHelper(params)
        helper.perform_fit()
    elif params.model.type == "ReRanker":
        helper = FitHelper(params)
        helper.perform_fit()

def predict(params):
    helper = PredictHelper(params)
    helper.perform_predict()

def eval(params):
    helper = EvalHelper(params)
    helper = helper.perform_eval()

@hydra.main(config_path="settings", config_name="settings.yaml", version_base=None)
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()

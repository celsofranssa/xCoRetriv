# activate venv and set Python path
source ~/projects/venvs/xCoRetriv/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoRetriv/

python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  model=RetrieverBERT \
  data=Wiki10-31k \
  data.folds=[0]
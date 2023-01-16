# activate venv and set Python path
source ~/projects/venvs/xCoRetriv/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoRetriv/

python main.py \
  tasks=[fit] \
  trainer.max_epochs=3 \
  trainer.patience=1 \
  trainer.precision=16 \
  model=BERT \
  data=Wiki10-31k \
  data.folds=[0]
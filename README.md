# xCoRetriv

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:celsofranssa/xCoRetriv.git

# change directory to project folder
cd xCoRetriv/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path
export PYTHONPATH=$PATHONPATH:<path-to-project-dir>/xCoRetriv/

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Datasets
Download the datasets from [kaggle](https://www.kaggle.com/datasets/celsofranssa/xCoFormer_EMTC-datasets):
```
kaggle datasets download celsofranssa/xCoFormer_EMTC-datasets -p resource/dataset/ --unzip
```
After downloading the datasets from it should be placed inside the `resources/datasets/` folder as shown below:

```
xCoFormer_EMTC/
|-- resources
|   |-- datasets
|   |   |-- EURLEX57K
|   |   |   |-- test.jsonl
|   |   |   |-- train.jsonl
|   |   |   `-- val.jsonl

```

### 3. Test Run
The following bash command fits the BERT encoder over EURLEX57K dataset using batch_size=64 and a single epoch.
```
python main.py tasks=[fit] model=BERT_TGT data=EURLEX57K data.batch_size=64 trainer.max_epochs=1
```
If all goes well the following output should be produced:
```
GPU available: True, used: True
[2020-12-31 13:44:42,967][lightning][INFO] - GPU available: True, used: True
TPU available: None, using: 0 TPU cores
[2020-12-31 13:44:42,967][lightning][INFO] - TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2020-12-31 13:44:42,967][lightning][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name         | Type                          | Params
------------------------------------------------------------
0 | text_encoder  | BERTEncoder                   | 109 M
1 | label_encoder | BERTEncoder                   | 109 M
2 | loss          | NPairLoss                     | 0     
3 | mrr           | MRRMetric                     | 0     
------------------------------------------------------------
91.0 M    Trainable params


Epoch 0: 100%|███████████████████████████████████████████████████████| 5199/5199 [13:06<00:00,  6.61it/s, loss=5.57, v_num=1, val_mrr=0.041, val_loss=5.54]
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288 [00:17<00:00, 16.83it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'m_test_mrr': tensor(0.0410),
 'm_val_mrr': tensor(0.0410),
 'test_mrr': tensor(0.0410),
 'val_loss': tensor(5.5390, device='cuda:0'),
 'val_mrr': tensor(0.0410)}
--------------------------------------------------------------------------------
```

### Troubleshoot
Pytorch 1.10.2 still has some issues with Nvidia CUDA 11.3 on Ubuntu 20.04. Therefore, it is recommended to install directly from the source.
```
pip uninstall torch
pip install torch==1.10.2 --extra-index-url https://download.pytorch.org/whl/cu113
```
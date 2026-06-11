# ECE228 Group 16

A remote sensing aware Transformer for bitemporal land cover change detection. It converts two satellite images into spatial tokens and temporal semantic tokens, reasons about cross time changes with a residual Transformer, and predicts a binary change map. It is trained with a hybrid cross entropy and Dice loss. See project_report for details.

## Environment

Python 3.11 with an NVIDIA GPU and CUDA 12.8.

```
conda create -n ece228 python=3.11
conda activate ece228
pip install -r requirements.txt
```

torch, torchvision, and torch-scatter must match your CUDA version. For CUDA 12.8 use the wheels from https://download.pytorch.org/whl/cu128 and https://data.pyg.org/whl/torch-2.11.0+cu128.html

## Demo

```
python demo.py --split demo --checkpoint_root . --project_name checkpoints
```

Predicted change maps are written to samples/predict. On the seven sample pairs the model reaches about F1 92 and IoU 85.

## Train

The datasets are LEVIR-CD, WHU-CD, and DSIFN-CD, with pretrained checkpoints in the checkpoints folder. Crop each dataset into 256 patches with folders A, B, label and a list folder, add its path in data_config.py, then run

```
python main_cd.py --data_name LEVIR --net_G ReViT --loss ce_dice --optimizer sgd --lr 0.01 --max_epochs 250 --batch_size 8
```

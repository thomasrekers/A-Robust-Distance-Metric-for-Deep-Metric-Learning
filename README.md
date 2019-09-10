# A-Robust-Distance-Metric-for-Deep-Metric-Learning

# Train the model first:
#### example to run the demp.py
!pip install fire

from google.colab import drive

drive.mount('/content/drive/')

import os
os.chdir('/content/drive/My Drive/where you store the code') # sample os.chdir('/content/drive/My Drive/app/metric_proj/')


!CUDA_VISIBLE_DEVICES=0 python demo.py --n_epochs 1 --data "./path to your data" --list_file './cars_list.txt' --save './model_P' --metric 'rM' --loss 'triplet'

# data should be arranged like: ./path to your data/train/xxxx.jpg

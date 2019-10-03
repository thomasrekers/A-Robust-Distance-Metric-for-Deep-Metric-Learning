# A-Robust-Distance-Metric-for-Deep-Metric-Learning

# Train the model:
#### example to run the demp.py
!pip install fire

from google.colab import drive

drive.mount('/content/drive/')

import os
os.chdir('/content/drive/My Drive/where you store the code') # sample os.chdir('/content/drive/My Drive/app/metric_proj/')

**to train with alexnet**
!CUDA_VISIBLE_DEVICES=0 python demo_pa.py --n_epochs 100 --data "./path to your data/" --fine_tune True  --lr 0.01 --T 0.0015 --model_name a_0.dat --result 'a_0.csv' --sampler 'semi' --lambda_ 0.00 --save './log/alex_tri_E' --metric 'E' --loss 'triplet'

**to train with resnet**
!CUDA_VISIBLE_DEVICES=0 python demo_p.py --n_epochs 100 --data "./path to your data/" --fine_tune True --lr 0.01 --T 0.0015 --model_name res_0.dat --result 'res_0.csv' --sampler 'semi' --lambda_ 0.000 --save './log/li_E_car_16' --metric 'E' --loss 'triplet'


# data should be arranged like: ./path to your data/train/xxxx.jpg

**to evaluate the network**
! python evaluate.py --architecture 'resnet' --score 'recall' --data './../cars' --labels_test './labels_test' --model_path './../t_4.dat' --K 2 --batchSize 600 --embSize 16 --loaderSize 200 --metricName 'euclidian'

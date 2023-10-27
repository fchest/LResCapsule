# MTCA-CapsNet
Code for paper: Emotion recognition from EEG based on multi-task learning with capsule network and attention mechanism

## About the paper
* Title: [Emotion recognition from EEG based on multi-task learning with capsule network and attention mechanism](https://doi.org/10.1016/j.compbiomed.2022.105303)
* Authors: Chang Li, Bin Wang, Silin Zhang, Yu Liu, Rencheng Song, Juan Cheng, Xun Chen
* Institution: Hefei University of Technology
* Published in: Computers in Biology and Medicine

## Instructions
* Before running the code, please download the DEAP dataset, unzip it and place it into the right directory.  Each .mat data file contains the EEG signals and consponding labels of a subject. There are 2 arrays in the file: **data** 
and **labels**. The shape of **data** is (40, 40, 8064). The shape of **label** is (40,4). 
* Please run the deap_pre_process.py to Load the origin .mat data file and transform it into .pkl file.
* Using Main.py to train and test the model (10-fold cross-validation), result of 10 folds will be saved in a .xls file.
* The DEAP dataset can be found [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap).
* The usage on DREAMER dataset is the same as above. The DREAMER dataset can be found [here](https://zenodo.org/record/546113/accessrequest).

## Requirements
+ Pyhton3.5
+ pytorch (1.4.0 version)
* If you have any questions, please contact binwang@mail.hfut.edu.cn

## Reference
* [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch)
* [ynulonger/ijcnn](https://github.com/ynulonger/ijcnn)

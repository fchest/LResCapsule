# Code for paper: Light-weight Residual Convolution-based Capsule Network for EEG Emotion Recognition

# Instructions
* On the DEAP dataset, each signal was segmented by using a 1s sliding window containing 128 samples for the preprocessed experimental signals, and 2400 samples (40 trials × 60 s) were obtained, with a final data dimension of 2400 × 128 × 32 for each subject. 
* Due to the length difference between stimulus signals in the DREAMER dataset, there is also an increase or decrease in the number of EEG samples captured. We used the same sliding window technique to obtain 3728 EEG samples, with a final data dimension of 3728 × 128 × 14 for each subject.
* The DEAP dataset can be found [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap). The DREAMER dataset can be found [here](https://zenodo.org/record/546113/accessrequest).

# Preprocess
* Please run the deap_pre_process.py to Load the origin .mat data file and transform it into .pkl file.
* Please run the dreamer_pre_process.py to Load the origin .mat data file and transform it into .pkl file.

# Run
* Using LResCapsule.py to train and test the model (10-fold cross-validation) 
* The server runs instructions in the background：nohup bash LResCapsule.sh>LResCapsule.log&

# Requirements
+ Pyhton3.8.13
+ pytorch (1.12.0 version)

## Reference
* [ynulonger/ijcnn](https://github.com/ynulonger/ijcnn)
* [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch)

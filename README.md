# ASD-crowd-counting

This is an unofficial implementation of the ICASSP 2019 paper [Adaptive Scenario Discovery for Crowd Counting](https://arxiv.org/abs/1812.02393) by PyTorch. Different with the paper, I added some data augmentation methods that turn out to be effective. The data augmentation methods reference from [this paper](https://arxiv.org/abs/1902.01115). 

## Prerequisites

Python: 3.5

PyTorch: 1.0.1

## Code structure

`density_map.py` To generate the density map. 

`data.py` Data preprocess and augmentation. 

`model.py` The structure of the network. 

`logger.py` Utility for logging on tensorboard. 

`train.py` To train the model. 

`eval.py` To evaluate the model. 


## Train

I have trained the model on ShanghaiTech part B. This is the training and testing logs on TensorBoard.

Average loss on training set
![](./logs/loss.png)

MAE on test set
![](./logs/mae.png)

MSE on test set
![](./logs/mse.png) 

## Results

MAE: 7.28 MSE: 11.85 on ShanghaiTech part B. 

Download checkpoint: [Google Drive](https://drive.google.com/open?id=1X3HzIdx5O6aU7QawQSHsntblpiTplpwh)


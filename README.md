# SPACE: Structured Compression and Sharing of Representational Space for Continual Learning

This repository contains code to accompany our IEEE Access paper on SPACE: Structured Compression and Sharing of Representational Space for Continual Learning. https://ieeexplore.ieee.org/document/9605653

The experiments with CIFAR-10/100 and 8-datasets in the paper can be reproduced by going to PCA_CL_github directory. The run.sh file contains the run parameters and finalized hyperparameters. They can be changed by editing the file and source the file to run the experiment using the following command : 

```python
source run.sh 
```
Trained models can be found at PCA_CL_github/PARAM/ direactory. To run the trained model, comment out the training command and uncomment the test command in run.sh file for the desired dataset and then source it. 

## Citation 
```
@ARTICLE{9605653,
  author={Saha, Gobinda and Garg, Isha and Ankit, Aayush and Roy, Kaushik},
  journal={IEEE Access}, 
  title={SPACE: Structured Compression and Sharing of Representational Space for Continual Learning}, 
  year={2021},
  volume={9},
  number={},
  pages={150480-150494},
  doi={10.1109/ACCESS.2021.3126027}}
```

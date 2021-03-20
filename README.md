# Structured Compression and Sharing of Representational Space for Continual Learning

This repository contains code to accompany arXiv submission on Structured Compression and Sharing of Representational Space for Continual Learning. https://arxiv.org/pdf/2001.08650.pdf

The experiments with CIFAR-10/100 and 8-datasets in the paper can be reproduced by going to PCA_CL_github directory. The run.sh file contains the run parameters and finalized hyperparameters. They can be changed by editing the file and source the file to run the experiment using the following command : 

```python
source run.sh 
```
Trained models can be found at PCA_CL_github/PARAM/ direactory. To run the trained model, comment out the training command and uncomment the test command in run.sh file for the desired dataset and then source it. 

## Citation 
```
@misc{saha2021space,
      title={SPACE: Structured Compression and Sharing of Representational Space for Continual Learning}, 
      author={Gobinda Saha and Isha Garg and Aayush Ankit and Kaushik Roy},
      year={2021},
      eprint={2001.08650},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# FL-Defender: Combating Targeted Attacks in Federated Learning.
This repository contains PyTorch implementation of the paper: FL-Defender: Combating Targeted Attacks in Federated Learning.

## Paper 

[FL-Defender: Combating Targeted Attacks in Federated Learning]

## Content
The repository contains one jupyter notebook for each benchmark which can be used to re-produce the experiments reported in the paper for that benchmark. The notebooks contain clear instructions on how to run the experiments. 

## Data sets
[MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) will be automatically downloaded.
However, [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) requires a manual download using this [link](https://drive.google.com/file/d/1X86CyTJW77a1CCkAFPvN6pqceN63q2Tx/view?usp=sharing). 
After downloading [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/), please save it as imdb.csv in the data folder inside the folder IMDB.


## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

[TensorFlow 2](https://www.tensorflow.org/)


## Results

### Robustness to label-flipping attacks

<img src="LF_results.PNG" width="100%">

*Robustness to the label-flipping attacks.* </br></br>

<img src="results/cifar10_mild_robust.PNG" width="100%">

*Robustness to the label-flipping attack on the CIFAR10-Mild benchmark.* </br></br>

<img src="results/imdb_robust.PNG" width="100%">

*Robustness to the label-flipping attack on the IMDB benchmark.* </br></br>

### Accuracy stability

<img src="results/mnist_extreme_stability.png" width="100%">

*Results of source class stability under the label-flipping attack on the MNIST-Extreme benchmark with 30% attackers.* </br></br>


<img src="results/cifar10_iid_stability.png" width="50%"><img src="results/cifar10_mild_stability.png" width="50%">

*Results of source class stability under the label-flipping attack with 30% attackers on the CIFAR10-IID benchmark (left), CIFAR10-Mild benchmark (right).*
</br></br>



## Citation 



## Funding


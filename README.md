# SGAAE

Python 3.8.3 and Pytorch 1.12.1 implementation of the latent Signed Archetypal Autoencoder (SGAA).

## Description

## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```

### Additional packages

Our Pytorch implementation uses the [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation guidelines can be found at the corresponding [Github repository](https://github.com/rusty1s/pytorch_sparse).

#### For a cpu installation please use: 

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html```

#### For a gpu installation please use:

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html```

where ${CUDA} should be replaced by either cu102, cu113, or cu116 depending on your PyTorch installation.



## Learning embeddings for signed undirected networks using SGAAE

**RUN:** &emsp; ```python main.py```

optional arguments:

**--epochs**  &emsp;  number of epochs for training (default: 3K)


**--cuda**  &emsp;    CUDA training (default: True)

**--LP**   &emsp;     performs link prediction (default: True)

**--D**   &emsp;      dimensionality of the embeddings (default: 8)

**--lr**   &emsp;     learning rate for the ADAM optimizer (default: 0.005)

**--dataset** &emsp;  dataset  (default: wiki_elec)

**--sample_percentage** &emsp;  sample size network percentage, it should be equal or less than 1 (default: 0.3)








# SGAAE

Python 3.8.3 and Pytorch 1.12.1 implementation of the Signed Graph Archetypal Autoencoder (SGAAE), as appeared in the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume 258. Copyright 2025 by the
author(s).

## Description

Autoencoders based on Graph Neural Networks (GNNs) have garnered significant attention in recent years for their ability to learn informative latent representations of complex topologies, such as graphs. Despite the prevalence of Graph Autoencoders, there has been limited focus on developing and evaluating explainable neural-based graph generative models specifically designed for signed networks. To address this gap, we propose the Signed Graph Archetypal Autoencoder (SGAAE) framework. SGAAE extracts node-level representations that express node memberships over distinct extreme profiles, referred to as archetypes, within the network. This is achieved by projecting the graph onto a learned polytope, which governs its polarization. The framework employs the Skellam distribution for analyzing signed networks combined with relational archetypal analysis and GNNs. Our experimental evaluation demonstrates the SGAAEs' capability to successfully infer node memberships over underlying latent structures while extracting competing communities. Additionally, we introduce the 2-level network polarization problem and show how SGAAE is able to characterize such a setting. The proposed model achieves high performance in different tasks of signed link prediction across four real-world datasets, outperforming several baseline models. Finally, SGAAE allows for interpretable visualizations in the polytope space, revealing the distinct aspects of the network, as well as, how nodes are expressing them.

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


## Reference

[Signed Graph Autoencoder for Explainable and Polarization-Aware Network Embeddings](https://www.arxiv.org/pdf/2409.10452). Nikolaos Nakis, Chrysoula Kosma, Giannis Nikolentzos, Michail Chatzianastasis, Iakovos Evdaimon, and Michalis Vazirgiannis, AISTATS 25






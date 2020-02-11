# Affective Attention
This repository contains source code for the ACL 2019 paper [Attention-based Conditioning Methods for External Knowledge Integration](https://www.aclweb.org/anthology/P19-1385/).

## Introduction
In this paper, we present a novel approach for incorporating external knowledge in Recurrent Neural Networks (RNNs). We propose the integration of lexicon features into the self-attention mechanism of RNN-based
architectures. This form of conditioning on the attention distribution, enforces the contribution of the most salient words for the task at hand. We introduce three methods, namely _attentional concatenation_, _feature-based gating_
and _affine transformation_. 

Experiments on six benchmark datasets show the effectiveness of our methods. Attentional feature-based gating
yields consistent performance improvement across tasks. Our approach is implemented as a simple add-on module for RNN-based models with minimal computational overhead and can be adapted to any deep neural architecture.

## Model 
We extend the standard self-attention mechanism, in order to condition the attention distribution of a given sentence, on each word’s prior lexical information.
Our methods, namely conditional concatenation, feature-base gating and affine transformation are depicted in the figure below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/28900064/74201901-e0c90d80-4c62-11ea-93ee-c0dfba7b8759.png" height="800">
</p>

## Reference
```
@inproceedings{margatina-etal-2019-attention,
    title = "Attention-based Conditioning Methods for External Knowledge Integration",
    author = "Margatina, Katerina  and Baziotis, Christos  and Potamianos, Alexandros",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1385",
    pages = "3944--3951"}
```

## Prerequisites

### Dependencies
 - PyTorch version >= 1.0.0
 - Python version >= 3.6
 
### Install Requirements

**Create Environment (Optional)**: Ideally, you should create an environment for the project.

```
conda create -n att_env python=3
conda activate att_env
```

Install PyTorch `1.0` with the desired Cuda version if you want to use the GPU
and then the rest of the requirements:

```
pip install -r requirements.txt
```

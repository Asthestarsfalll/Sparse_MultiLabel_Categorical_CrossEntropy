# Sparse Multilabel Categorical CrossEntropy

## Introduction

This repo contains Sparse Multilabel Categorical CrossEntropy(`SMCCE`) functions implemented by PyTorch, MegEngine and Paddle.

`SMCCE`  is the sparse version of Multilabel Categorical CrossEntropy(`MCCE`). When positive examples are much less than negative examples, it's able to significantly reduce the size of the label matrix and speed up the training without performance loss.

More detail information :  https://kexue.fm/archives/8888, https://juejin.cn/post/7064063040441810957

Please note that this implementation does not need change the num_classes to num_classes + 1, however original implementation needs.

You should remove `[CLS]` token before call this function. 

## Usage

```python
from SMCCE_pytorch import sparse_multilabel_categorical_crossentropy as smcce

def loss_fn(label, pred):
    # if need, process the label and pred to demanded format
    # ...
    return smcce(label, pred)


# ...
# your training codes....
# ...


loss = loss_fn(label, pred)
```

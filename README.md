

## Introduction

This repo contains Sparse Multilabel Categorical CrossEntropy(`SMCCE`) functions implemented by PyTorch and MegEngine.

`SMCCE`  is the sparse version of Multilabel Categorical CrossEntropy(`MCCE`). When positive examples are much less than negative examples, it's able to significantly reduce the size of the label matrix and speed up the training without performance loss.

More detial information :  https://kexue.fm/archives/8888

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


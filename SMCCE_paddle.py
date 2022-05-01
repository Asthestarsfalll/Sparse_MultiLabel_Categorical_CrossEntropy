import paddle
import numpy as np
import paddle.nn.functional as F
from paddle import Tensor


def batch_gather(input: Tensor, indices: Tensor):
    """
    Args:
        input: label tensor with shape [batch_size, n, L] or [batch_size, L]
        indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]

    Return:
        Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input 
        if m == n, the return shape is [batch_size, m, l]
        if m != n, the return shape is [batch_size, n, l*m]

    """
    results = []
    for data, index in zip(input, indices):
        if len(index) < len(data):
            index = index.reshape([-1])
            print(index.shape)
            results.append(paddle.index_select(data, index, axis=-1))
        else:
            indice_dim = index.ndim
            results.append(paddle.take_along_axis(
                data, axis=indice_dim-1, indices=index))
    return paddle.stack(results)


def sparse_multilabel_categorical_crossentropy(label: Tensor, pred: Tensor, mask_zero=False, reduction='none'):
    """Sparse Multilabel Categorical CrossEntropy
        Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272

    Args:
        label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
            should contain the indexes of the positive rather than a ont-hot vector.
        pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
        mask_zero: if label is used zero padding to align, please specify make_zero=True.
            when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.

    """
    zeros = paddle.zeros_like(pred[..., :1])
    pred = paddle.concat([pred, zeros], axis=-1)
    if mask_zero:
        infs = paddle.ones_like(zeros) * np.nan
        pred = paddle.concat([infs, pred], axis=-1)
    pos_2 = batch_gather(pred, label)
    pos_1 = paddle.concat([pos_2, zeros], axis=-1)
    if mask_zero:
        pred = paddle.concat([-infs, pred], axis=-1)
        pos_2 = batch_gather(pred, label)
    pos_loss = paddle.logsumexp(-pos_1, axis=-1)
    all_loss = paddle.logsumexp(pred, axis=-1)
    aux_loss = paddle.logsumexp(pos_2, axis=-1) - all_loss
    aux_loss = paddle.clip(1 - F.exp(aux_loss), 1e-16, 1)
    neg_loss = all_loss + paddle.log(aux_loss)
    loss = pos_loss + neg_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


if __name__ == '__main__':
    x = paddle.to_tensor(np.arange(384).reshape(2, 3, 64))
    y = paddle.to_tensor(np.arange(1024).reshape(2, 8, 64))
    indices = paddle.to_tensor(
        np.array(
            [
                [[1, 2, 3, 4], [0, 1, 0, 0], [0, 0, 0, 0]],
                [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ]
        ))
    print(indices.shape)
    print('='*80)
    res = batch_gather(x, indices)
    print(res.shape)
    print(res)
    print('='*80)
    res = batch_gather(y, indices)
    print(res.shape)
    print(res)

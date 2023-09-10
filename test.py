import torch

from SMCCE_pytorch import sparse_multilabel_categorical_crossentropy

pred = torch.randn((8, 512))
label = torch.randint(0, 512, (8, 4))
loss = sparse_multilabel_categorical_crossentropy(pred, label, mask_zero=True, reduction='mean')
print(loss)

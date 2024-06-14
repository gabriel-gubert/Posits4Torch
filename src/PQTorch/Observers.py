import torch
from torch.ao.quantization import ObserverBase

class LazyObserver(ObserverBase):
    def __init__(self, dtype):
        super().__init__(dtype)

    def forward(self, X):
        return X

    def calculate_qparams(self, **kwargs):
        return 1.0, 0

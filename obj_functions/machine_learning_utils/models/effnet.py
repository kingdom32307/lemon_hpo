import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from models.base_model import BaseModel

class EffNet(BaseModel):
    def __init__(self, num_class, pre_trained=False, freeze_weight=False):
        super().__init__()
        if pre_trained:
            self.base = EfficientNet.from_pretrained(self._model_name)
        else:
            self.base = EfficientNet.from_name(self._model_name)
        if freeze_weight:
            for param in self.base.parameters():
                param.requires_grad = False

        num_ftrs = self.base._fc.in_features
        self.base._fc = nn.Linear(num_ftrs, num_class)

    def forward(self, x):
        out = self.base(x)
        return out

class EffNet0(EffNet):
    def __init__(self, num_class, pre_trained=False, freeze_weight=False):
        self._model_name = 'efficientnet-b0'
        super().__init__(num_class=num_class, pre_trained=pre_trained,
            freeze_weight=freeze_weight)

class EffNet1(EffNet):
    def __init__(self, num_class, pre_trained=False, freeze_weight=False):
        self._model_name = 'efficientnet-b1'
        super().__init__(num_class=num_class, pre_trained=pre_trained,
            freeze_weight=freeze_weight)

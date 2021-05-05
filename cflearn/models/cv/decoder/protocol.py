import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from cftool.misc import shallow_copy_dict

from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import WithRegister
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY


decoders: Dict[str, Type["DecoderBase"]] = {}


class DecoderBase(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DecoderBase"]] = decoders

    def __init__(
        self,
        img_size: int,
        latent_channels: int,
        latent_resolution: int,
        num_upsample: int,
        out_channels: int,
        *,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_channels = latent_channels
        self.latent_resolution = latent_resolution
        self.num_upsample = num_upsample
        self.out_channels = out_channels
        # conditional
        self.cond_channels = cond_channels
        self.num_classes = num_classes
        self.cond = None
        if num_classes is not None:
            shape = num_classes, cond_channels, latent_resolution, latent_resolution
            self.cond = nn.Parameter(torch.empty(*shape))
            with torch.no_grad():
                nn.init.zeros_(self.cond.data)

    @property
    def is_conditional(self) -> bool:
        return self.num_classes is not None

    def _inject_cond(self, batch: tensor_dict_type) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        if self.cond is not None:
            cond = self.cond[batch[LABEL_KEY].view(-1)]
            batch[INPUT_KEY] = torch.cat([batch[INPUT_KEY], cond], dim=1)
        return batch

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def decode(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        return self.forward(0, batch, **kwargs)


__all__ = ["DecoderBase"]

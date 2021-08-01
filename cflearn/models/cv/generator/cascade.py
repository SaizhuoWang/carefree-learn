import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import shallow_copy_dict

from ...bases import CascadeBase
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import imagenet_normalize


@ModelProtocol.register("cascade_u2net")
class CascadeU2Net(CascadeBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        lv1_model_ckpt_path: str,
        lv2_model_config: Optional[Dict[str, Any]] = None,
        latent_channels: int = 32,
        num_layers: int = 5,
        max_layers: int = 7,
        lite: bool = False,
    ):
        super().__init__()
        lv1_model_config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            max_layers=max_layers,
            lite=lite,
        )
        if lv2_model_config is None:
            lv2_model_config = shallow_copy_dict(lv1_model_config)
        lv2_model_config["in_channels"] = in_channels * 2
        self._construct(
            "u2net",
            "u2net",
            lv1_model_config,
            lv2_model_config,
            lv1_model_ckpt_path,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        lv1_outputs = self.lv1_net(batch_idx, batch, state, **kwargs)
        lv1_logits = lv1_outputs[PREDICTIONS_KEY]
        lv1_alpha = torch.sigmoid(lv1_logits[0])
        net = batch[INPUT_KEY]
        masked_net = net * lv1_alpha
        lv2_input = torch.cat([net, masked_net], dim=1)
        lv2_outputs = self.lv2_net(batch_idx, {INPUT_KEY: lv2_input}, state, **kwargs)
        lv2_logits = lv2_outputs[PREDICTIONS_KEY]
        merged_logits = [lv1 + lv2 for lv1, lv2 in zip(lv1_logits, lv2_logits)]
        return {PREDICTIONS_KEY: merged_logits}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY][0]


__all__ = [
    "CascadeU2Net",
]

import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..encoder import ViTEncoder
from ..encoder import Encoder1DBase
from ..encoder import Encoder1DFromPatches
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import INFO_PREFIX
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import download_model
from ....modules.blocks import Linear


@ModelProtocol.register("clf")
class VanillaClassifier(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Optional[int] = None,
        latent_dim: int = 128,
        aux_num_classes: Optional[Dict[str, int]] = None,
        *,
        encoder1d: str = "vanilla",
        encoder1d_config: Optional[Dict[str, Any]] = None,
        encoder1d_pretrained_name: Optional[str] = None,
        encoder1d_pretrained_path: Optional[str] = None,
        encoder1d_pretrained_strict: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        # encoder1d
        if encoder1d_config is None:
            encoder1d_config = {}
        if Encoder1DFromPatches.check_subclass(encoder1d):
            encoder1d_config["img_size"] = img_size
        if ViTEncoder.check_subclass(encoder1d) and aux_num_classes is not None:
            encoder1d_config["aux_heads"] = sorted(aux_num_classes)
        encoder1d_config["in_channels"] = in_channels
        encoder1d_config["latent_dim"] = latent_dim
        self.encoder1d = Encoder1DBase.make(encoder1d, config=encoder1d_config)
        if encoder1d_pretrained_path is None:
            if encoder1d_pretrained_name is not None:
                encoder1d_pretrained_path = download_model(encoder1d_pretrained_name)
        if encoder1d_pretrained_path is not None:
            print(
                f"{INFO_PREFIX}loading pretrained encoder1d "
                f"from '{encoder1d_pretrained_path}'"
            )
            d = torch.load(encoder1d_pretrained_path)
            self.encoder1d.load_state_dict(d, strict=encoder1d_pretrained_strict)
        # head
        main_head = Linear(latent_dim, num_classes)
        self.head = None
        self.aux_keys = None
        if aux_num_classes is None:
            self.head = main_head
        else:
            heads = {LATENT_KEY: main_head}
            self.aux_keys = []
            for key, n in aux_num_classes.items():
                heads[key] = Linear(latent_dim, n)
                self.aux_keys.append(key)
            self.heads = torch.nn.ModuleDict(heads)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        *,
        return_encodings: bool = False,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        encodings = self.encoder1d(batch_idx, batch, state, **kwargs)
        if return_encodings:
            return encodings
        if self.head is not None:
            return {PREDICTIONS_KEY: self.head(encodings[LATENT_KEY])}
        results = {}
        for key, head in self.heads.items():
            predictions = head(encodings[key])
            results[PREDICTIONS_KEY if key == LATENT_KEY else key] = predictions
        return results

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.classify(batch[INPUT_KEY])

    def classify(self, net: Tensor, **kwargs: Any) -> Union[Tensor, tensor_dict_type]:
        rs = self.forward(0, {INPUT_KEY: net}, **kwargs)
        if self.aux_keys is None:
            return rs[PREDICTIONS_KEY]
        return {key: rs[key] for key in [PREDICTIONS_KEY] + self.aux_keys}


__all__ = ["VanillaClassifier"]

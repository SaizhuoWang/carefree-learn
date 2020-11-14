import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from typing import Optional
from cftool.misc import register_core
from cftool.misc import LoggingMixin
from cfdata.tabular import TabularData

from ..transform import Dimensions
from ...misc.configs import configs_dict
from ...misc.configs import Configs


head_dict: Dict[str, Type["HeadBase"]] = {}


class HeadConfigs(Configs):
    def __init__(
        self,
        in_dim: int,
        tr_data: TabularData,
        dimensions: Dimensions,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.in_dim = in_dim
        self.tr_data = tr_data
        self.dimensions = dimensions

    @property
    def out_dim(self) -> int:
        out_dim: int = self.config.get("out_dim")
        default_out_dim = max(self.tr_data.num_classes, 1)
        if out_dim is None:
            out_dim = default_out_dim
        return out_dim

    def pop(self) -> Dict[str, Any]:
        config = super().pop()
        config["in_dim"] = self.in_dim
        config["out_dim"] = self.out_dim
        return config

    @classmethod
    def get(
        cls,
        scope: str,
        name: str,
        *,
        in_dim: Optional[int] = None,
        tr_data: Optional[TabularData] = None,
        dimensions: Optional[Dimensions] = None,
        **kwargs: Any,
    ) -> "HeadConfigs":
        if in_dim is None:
            raise ValueError("`in_dim` must be provided for `HeadConfigs`")
        if tr_data is None:
            raise ValueError("`tr_data` must be provided for `HeadConfigs`")
        if dimensions is None:
            raise ValueError("`dimensions` must be provided for `HeadConfigs`")
        cfg_type = configs_dict[scope][name]
        if not issubclass(cfg_type, HeadConfigs):
            raise ValueError(f"'{name}' under '{scope}' scope is not `HeadConfigs`")
        return cfg_type(in_dim, tr_data, dimensions, kwargs)


class HeadBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        pass

    def _init_config(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global head_dict
        return register_core(name, head_dict)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "HeadBase":
        return head_dict[name](**config)


__all__ = ["HeadConfigs", "HeadBase"]

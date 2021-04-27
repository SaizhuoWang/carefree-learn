import torch

from typing import Any
from typing import Optional

from .data import MLLoader
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...protocol import InferenceOutputs
from ...protocol import InferenceProtocol


class MLInference(InferenceProtocol):
    def get_outputs(  # type: ignore
        self,
        loader: MLLoader,
        device: Optional[torch.device] = None,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        loss: Optional[LossProtocol] = None,
        return_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        kwargs["loader_name"] = loader.name
        return super().get_outputs(
            loader,
            device,
            portion=portion,
            state=state,
            loss=loss,
            return_outputs=return_outputs,
            use_tqdm=use_tqdm,
            **kwargs,
        )


class DLInference(InferenceProtocol):
    pass


__all__ = [
    "MLInference",
    "DLInference",
]

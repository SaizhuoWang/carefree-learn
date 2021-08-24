import torch
import random

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from torch.optim import Optimizer

from .losses import GANLoss
from .losses import GANTarget
from .discriminators import DiscriminatorBase
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....protocol import StepOutputs
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....protocol import ModelWithCustomSteps
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import to_device
from ....misc.toolkit import mode_context
from ....misc.internal_ import CVLoader


class GANMixin(ModelWithCustomSteps, GaussianGeneratorMixin, metaclass=ABCMeta):
    def __init__(
        self,
        img_size: int,
        *,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.gan_mode = gan_mode
        self.gan_loss = GANLoss(gan_mode)
        if gan_loss_configs is None:
            gan_loss_configs = {}
        self.lambda_gp = gan_loss_configs.get("lambda_gp", 10.0)

    @property
    @abstractmethod
    def g_parameters(self) -> List[nn.Parameter]:
        pass

    @property
    @abstractmethod
    def d_parameters(self) -> List[nn.Parameter]:
        pass

    @abstractmethod
    def _g_loss(
        self,
        batch: tensor_dict_type,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # loss_g, sampled, labels
        pass

    @abstractmethod
    def _d_losses(
        self,
        net: Tensor,
        sampled: Tensor,
        labels: Optional[Tensor],
    ) -> Tuple[Tensor, tensor_dict_type]:
        # d_loss, losses
        pass

    # utilities

    @property
    def can_reconstruct(self) -> bool:
        return False

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z = torch.randn(len(batch[INPUT_KEY]), self.latent_dim, device=self.device)
        return {PREDICTIONS_KEY: self.decode(z, labels=batch[LABEL_KEY], **kwargs)}

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self._g_loss(batch, {})

    @staticmethod
    def _gather_losses(loss_g: Tensor, loss_dict: tensor_dict_type) -> None:
        loss_dict["g_gan"] = loss_g
        loss = sum([sub_loss.detach() for sub_loss in loss_dict.values()])
        loss_dict[LOSS_KEY] = loss

    def _toggle_optimizer(self, optimizer: Optimizer) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = True


class VanillaGANMixin(GANMixin, metaclass=ABCMeta):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        *,
        discriminator: str = "basic",
        discriminator_configs: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            img_size,
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_configs=gan_loss_configs,
        )
        if discriminator_configs is None:
            discriminator_configs = {}
        discriminator_configs["img_size"] = img_size
        discriminator_configs["in_channels"] = in_channels
        discriminator_configs["num_classes"] = num_classes
        self.discriminator = DiscriminatorBase.make(
            discriminator,
            config=discriminator_configs,
        )

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def _g_loss(
        self,
        batch: tensor_dict_type,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled = self.sample(len(batch[INPUT_KEY]), labels=labels, **forward_kwargs)
        pred_fake = self.discriminator(sampled)
        loss_g = self.gan_loss(pred_fake, GANTarget(True, labels))
        return loss_g, sampled, labels

    def _d_losses(
        self,
        net: Tensor,
        sampled: Tensor,
        labels: Optional[Tensor],
    ) -> Tuple[Tensor, tensor_dict_type]:
        pred_real = self.discriminator(net)
        loss_d_real = self.gan_loss(pred_real, GANTarget(True, labels))
        pred_fake = self.discriminator(sampled)
        loss_d_fake = self.gan_loss(pred_fake, GANTarget(False, labels))
        d_loss = 0.5 * (loss_d_fake + loss_d_real)
        losses = {"d_fake": loss_d_fake, "d_real": loss_d_real}
        if self.gan_mode == "wgangp":
            eps = random.random()
            merged = eps * net + (1 - eps) * sampled
            with mode_context(self.discriminator, to_train=None, use_grad=True):
                pred_merged = self.discriminator(merged.requires_grad_(True)).output
                loss_gp = self.gan_loss.loss(merged, pred_merged)
            d_loss = d_loss + self.lambda_gp * loss_gp
            losses["d_gp"] = loss_gp
        return d_loss, losses

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        opt_g = trainer.optimizers["g_parameters"]
        opt_d = trainer.optimizers["d_parameters"]
        # generator step
        self._toggle_optimizer(opt_g)
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            loss_g, sampled, labels = self._g_loss(batch, forward_kwargs)
        trainer.grad_scaler.scale(loss_g).backward()
        if trainer.clip_norm > 0.0:
            trainer._clip_norm_step()
        trainer.grad_scaler.step(opt_g)
        trainer.grad_scaler.update()
        opt_g.zero_grad()
        # discriminator step
        self._toggle_optimizer(opt_d)
        with torch.no_grad():
            sampled = sampled.detach().clone()
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            loss_d, loss_dict = self._d_losses(batch[INPUT_KEY], sampled, labels)
        trainer.grad_scaler.scale(loss_d).backward()
        if trainer.clip_norm > 0.0:
            trainer._clip_norm_step()
        trainer.grad_scaler.step(opt_d)
        trainer.grad_scaler.update()
        opt_d.zero_grad()
        # finalize
        trainer._scheduler_step()
        forward_results = {PREDICTIONS_KEY: sampled}
        self._gather_losses(loss_g, loss_dict)
        return StepOutputs(forward_results, loss_dict)

    def evaluate_step(  # type: ignore
        self,
        loader: CVLoader,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        loss_items: Dict[str, List[float]] = {}
        for i, batch in enumerate(loader):
            if i / len(loader) >= portion:
                break
            batch = to_device(batch, self.device)
            loss_g, sampled, labels = self._g_loss(batch, {})
            loss_d, loss_dict = self._d_losses(batch[INPUT_KEY], sampled, labels)
            self._gather_losses(loss_g, loss_dict)
            for k, v in loss_dict.items():
                loss_items.setdefault(k, []).append(v.item())
        # gather
        mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
        score = trainer._weighted_loss_score(mean_loss_items)
        return MetricsOutputs(score, mean_loss_items)


__all__ = [
    "GANMixin",
    "VanillaGANMixin",
]

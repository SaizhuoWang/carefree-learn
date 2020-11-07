import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import context_error_handler

from ...types import tensor_tuple_type
from ...misc.toolkit import switch_requires_grad
from ...misc.toolkit import Activations
from ...modules.blocks import MLP
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import MonotonousMapping
from ...modules.blocks import ConditionalBlocks
from ...modules.blocks import ConditionalOutput
from ...modules.blocks import PseudoInvertibleBlock


def default_transition_builder(dim: int) -> nn.Module:
    h_dim = int(dim // 2)
    return MonotonousMapping.make_couple(h_dim, h_dim, h_dim, "tanh", ascent=True)


def monotonous_builder(
    ascent1: bool,
    ascent2: bool,
    to_latent: bool,
    num_layers: int,
    condition_dim: int,
) -> Callable[[int, int], nn.Module]:
    def _core(in_dim: int, out_dim: int, ascent: bool) -> ConditionalBlocks:
        cond_out_dim: Optional[int]
        block_out_dim: Optional[int]
        if to_latent:
            num_units = [out_dim] * (num_layers + 1)
            cond_out_dim = block_out_dim = None
        else:
            num_units = [in_dim] * num_layers
            if out_dim == 1:
                cond_out_dim = block_out_dim = 1
            else:
                # quantile stuffs
                # cond  : median, pos_median_res, neg_median_res
                # block : y_pos_add, y_pos_mul, y_neg_add, y_neg_mul
                cond_out_dim = out_dim
                block_out_dim = out_dim + 1

        blocks = MonotonousMapping.stack(
            in_dim,
            block_out_dim,
            num_units,
            ascent=ascent,
            return_blocks=True,
        )
        assert isinstance(blocks, list)

        cond_module = MLP.simple(
            condition_dim,
            cond_out_dim,
            num_units,
            activation="mish",
        )
        cond_mappings = cond_module.mappings

        return ConditionalBlocks(
            nn.ModuleList(blocks),
            cond_mappings,
            add_last=to_latent,
        )

    def _split_core(in_dim: int, out_dim: int) -> nn.Module:
        if not to_latent:
            in_dim = int(in_dim // 2)
        else:
            out_dim = int(out_dim // 2)

        class MonoSplit(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = _core(in_dim, out_dim, ascent1)
                self.m2 = _core(in_dim, out_dim, ascent2)

            def forward(
                self,
                net: Union[Tensor, tensor_tuple_type],
                cond: Tensor,
            ) -> Union[Tensor, tensor_tuple_type, ConditionalOutput]:
                if to_latent:
                    assert isinstance(net, Tensor)
                    return self.m1(net, cond), self.m2(net, cond)
                assert isinstance(net, tuple)
                o1, o2 = self.m1(net[0], cond), self.m2(net[1], cond)
                return ConditionalOutput(o1.net + o2.net, o1.cond + o2.cond)

        return MonoSplit()

    return _split_core


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        y_min: float,
        y_max: float,
        fetch_q: bool,
        fetch_cdf: bool,
        num_layers: Optional[int] = None,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
        transition_builder: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        # common
        self.y_min = y_min
        self.y_diff = y_max - y_min
        if not fetch_q and not fetch_cdf:
            raise ValueError("something must be fetched, either `q` or `cdf`")
        self.fetch_q = fetch_q
        self.fetch_cdf = fetch_cdf
        self.mish = Activations().mish
        if num_layers is None:
            num_layers = 1
        if num_blocks is None:
            num_blocks = 2
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        if latent_dim is None:
            latent_dim = 512
        if transition_builder is None:
            transition_builder = default_transition_builder
        # pseudo invertible q
        kwargs = {"num_layers": num_layers, "condition_dim": in_dim}
        if not self.fetch_q:
            q_to_latent_builder = self.dummy_builder
        else:
            q_to_latent_builder = monotonous_builder(True, True, True, **kwargs)
        if not self.fetch_cdf:
            q_from_latent_builder = self.dummy_builder
        else:
            q_from_latent_builder = monotonous_builder(True, False, False, **kwargs)
        self.q_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            1,
            to_transition_builder=q_to_latent_builder,
            from_transition_builder=q_from_latent_builder,
        )
        # pseudo invertible y
        if not self.fetch_cdf:
            y_to_latent_builder = self.dummy_builder
        else:
            y_to_latent_builder = monotonous_builder(True, False, True, **kwargs)
        if not self.fetch_q:
            y_from_latent_builder = self.dummy_builder
        else:
            y_from_latent_builder = monotonous_builder(True, True, False, **kwargs)
        self.y_invertible = PseudoInvertibleBlock(
            1,
            latent_dim,
            3,
            to_transition_builder=y_to_latent_builder,
            from_transition_builder=y_from_latent_builder,
        )
        # q parameters
        if not self.fetch_q:
            self.q_parameters = []
        else:
            q_params1 = list(self.q_invertible.to_latent.parameters())
            q_params2 = list(self.y_invertible.from_latent.parameters())
            self.q_parameters = q_params1 + q_params2
        # invertible blocks
        self.num_blocks = num_blocks
        self.block_parameters: List[nn.Parameter] = []
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, transition_builder=transition_builder)
            self.block_parameters.extend(block.parameters())
            self.blocks.append(block)

    @property
    def q_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def y_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda y: (y - self.y_min) / (0.5 * self.y_diff) - 1.0

    @property
    def q_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return torch.sigmoid

    @property
    def y_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda y: (y + 1.0) * (0.5 * self.y_diff) + self.y_min

    @property
    def dummy_builder(self) -> Callable[[int, int], nn.Module]:
        return lambda _, __: nn.Identity()

    def _detach_q(self) -> context_error_handler:
        def switch(requires_grad: bool) -> None:
            switch_requires_grad(self.q_parameters, requires_grad)
            switch_requires_grad(self.block_parameters, requires_grad)

        class _(context_error_handler):
            def __enter__(self) -> None:
                switch(False)

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                switch(True)

        return _()

    def _merge_q_outputs(
        self,
        outputs: ConditionalOutput,
        q_batch: Tensor,
        median: bool,
    ) -> Dict[str, Optional[Tensor]]:
        y_net = outputs.net
        cond_net = outputs.cond
        y_split = y_net.split(1, dim=1)
        cond_split = cond_net.split(1, dim=1)
        med, pos_med_res, neg_med_res = cond_split
        y_pos_add, y_pos_mul, y_neg_add, y_neg_mul = y_split
        pos_med_res = self.mish(pos_med_res)
        neg_med_res = -self.mish(neg_med_res)
        y_pos_mul = y_pos_mul.relu_()
        y_neg_mul = (1.0 - y_neg_mul).relu_()
        if median:
            q_sign = q_positive_mask = None
            y_res = add_net = mul_net = med_res = None
        else:
            q_sign = torch.sign(q_batch)
            q_positive_mask = q_sign == 1.0
            add_net = torch.where(q_positive_mask, y_pos_add, y_neg_add)
            mul_net = torch.where(q_positive_mask, y_pos_mul, y_neg_mul)
            med_res = torch.where(q_positive_mask, pos_med_res, neg_med_res)
            y_res = med_res * mul_net + add_net
        results = {"median": med}
        if median:
            results.update(
                {
                    "pos_add": y_pos_add,
                    "neg_add": y_neg_add,
                    "pos_mul": y_pos_mul,
                    "neg_mul": y_neg_mul,
                }
            )
        else:
            results.update(
                {
                    "y_res": y_res,
                    "med_add": add_net,
                    "med_mul": mul_net,
                    "med_res": med_res,
                    "q_sign": q_sign,
                    "q_positive_mask": q_positive_mask,
                }
            )
        return results

    def _q_results(
        self,
        net: Tensor,
        q_batch: Optional[Tensor] = None,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        # prepare q_latent
        if q_batch is None and median:
            if q_batch is not None:
                msg = "`median` is specified but `q_batch` is still provided"
                raise ValueError(msg)
            q_batch = net.new_zeros(len(net), 1)
        if q_batch is None:
            q1 = q2 = None
        else:
            if not median:
                q_batch = self.q_fn(q_batch)
            q1, q2 = self.q_invertible(q_batch, net)
            q1, q2 = q1.net, q2.net
        # simulate quantile function
        q_inverse = None
        if q_batch is None:
            y_results = None
        else:
            assert q1 is not None and q2 is not None
            for block in self.blocks:
                q1, q2 = block(q1, q2)
            y_pack = self.y_invertible.inverse((q1, q2), net)
            assert isinstance(y_pack, ConditionalOutput)
            y_results = self._merge_q_outputs(y_pack, q_batch, median)
            if do_inverse and self.fetch_cdf:
                median_output, y_res = map(y_results.get, ["median", "y_res"])
                assert median_output is not None
                y = median_output.detach()
                if not median:
                    assert y_res is not None
                    y = y + y_res.detach()
                inverse_results = self._y_results(net, y)
                q_inverse = inverse_results["q"]
        results: Dict[str, Optional[Tensor]] = {}
        if y_results is not None:
            results.update(y_results)
        results["q_inverse"] = q_inverse
        return results

    def _y_results(
        self,
        net: Tensor,
        y_batch: Optional[Tensor] = None,
        do_inverse: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        # prepare y_latent
        if y_batch is None:
            y1 = y2 = None
        else:
            y_batch = self.y_fn(y_batch)
            y1, y2 = self.y_invertible(y_batch, net)
            y1, y2 = y1.net, y2.net
        # simulate cdf
        y_inverse_res = None
        if y_batch is None:
            q = q_logit = None
        else:
            for i in range(self.num_blocks):
                y1, y2 = self.blocks[self.num_blocks - i - 1].inverse(y1, y2)
            q_logit_pack = self.q_invertible.inverse((y1, y2), net)
            assert isinstance(q_logit_pack, ConditionalOutput)
            q_logit = q_logit_pack.net
            q = self.q_inv_fn(q_logit)
            if do_inverse and self.fetch_q:
                with self._detach_q():
                    inverse_results = self._q_results(net, q.detach())
                    y_inverse_res = inverse_results["y_res"]
        return {"q": q, "q_logit": q_logit, "y_inverse_res": y_inverse_res}

    def forward(
        self,
        net: Tensor,
        *,
        q_batch: Optional[Tensor] = None,
        y_batch: Optional[Tensor] = None,
        do_inverse: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        results: Dict[str, Optional[Tensor]] = {}
        if self.fetch_q:
            results.update(self._q_results(net, q_batch, do_inverse, median))
        if self.fetch_cdf:
            results.update(self._y_results(net, y_batch, do_inverse))
        return results


__all__ = ["DDRCore"]

import copy
from typing import Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.regularisers import Regulariser
from ....models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from ....torch_utility import TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase
from .utility import DiscreteQFunctionMixin

__all__ = ["DQNImpl", "DoubleDQNImpl"]


class DQNImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _gamma: float
    _q_func: EnsembleDiscreteQFunction
    _targ_q_func: EnsembleDiscreteQFunction
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        optim: Optimizer,
        gamma: float,
        device: str,
        regulariser: Regulariser,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._gamma = gamma
        self._q_func = q_func
        self._optim = optim
        self._targ_q_func = copy.deepcopy(q_func)
        self._regulariser = regulariser

    @train_api
    def update(self, batch: TorchMiniBatch) -> np.array:
        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss, reg_val = self.compute_loss(batch, q_tpn)

        loss.backward()
        self._optim.step()

        res = np.array(
            [
                float(loss.cpu().detach().numpy()),
                float(reg_val.cpu().detach().numpy()),
            ]
        )
        return res

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        reg_val = self._regulariser(algo=self, batch=batch)
        return loss + reg_val, reg_val

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_func(x).argmax(dim=1)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def q_function(self) -> EnsembleQFunction:
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        return self._optim


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )

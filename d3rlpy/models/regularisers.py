import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Union

import torch

from ..algos.qlearning.base import QLearningAlgoImplBase
from ..serializable_config import DynamicConfig, generate_config_registration
from ..torch_utility import TorchMiniBatch
from .torch import Encoder, EncoderWithAction


class Regulariser(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self, algo: QLearningAlgoImplBase, batch: TorchMiniBatch
    ) -> torch.Tensor:
        pass


class DefaultRegulariser(Regulariser):
    """Basic regulariser class to use as default i.e., no regularisation is
    applied
    """

    def __call__(
        self, algo: QLearningAlgoImplBase, batch: TorchMiniBatch
    ) -> torch.Tensor:
        res = torch.tensor(0.0)
        return res


class _Dr3(Regulariser):
    """DR3 Regularisation for approximate dynamic programming

    References:
        * `Kumar et al., DR3: VALUE-BASED DEEP REINFORCEMENT LEARNING REQUIRES
        EXPLICIT REGULARIZATION. <https://openreview.net/pdf?id=POvMvLi91f>`_
    """

    @abstractmethod
    def get_features(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        observations: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def __call__(
        self, algo: QLearningAlgoImplBase, batch: TorchMiniBatch
    ) -> torch.Tensor:
        cur_features = []
        for q_func in algo.q_function._q_funcs:
            cur_features.append(
                self.get_features(
                    encoder=q_func.encoder,
                    observations=batch.observations,
                    action=batch.actions,
                )[None, :]
            )
        next_features = []
        for q_func in algo.q_function._q_funcs:
            next_features.append(
                self.get_features(
                    encoder=q_func.encoder,
                    observations=batch.next_observations,
                    action=algo.predict_best_action(x=batch.next_observations),
                )[None, :]
            )
        cur_features = torch.concat(cur_features).mean(dim=0)
        next_features = torch.concat(next_features).mean(dim=0)
        dot = torch.mul(cur_features, next_features).sum()
        return dot


class Dr3WithAction(_Dr3):
    """Dr3 regularisation to use with continuous encoders"""

    def get_features(
        self,
        encoder: EncoderWithAction,
        observations: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return encoder(x=observations, action=action)


class Dr3(_Dr3):
    """Dr3 regularisation to use with discrete encoders"""

    def get_features(
        self, encoder: Encoder, observations: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return encoder(x=observations)


@dataclasses.dataclass()
class RegulariserFactory(DynamicConfig):
    """Base regulariser factory class
    """

    def create_with_action(
        self,
    ) -> Regulariser:
        raise NotImplementedError()

    def create(
        self,
    ) -> Regulariser:
        raise NotImplementedError()


@dataclasses.dataclass()
class Dr3RegulariserFactory(RegulariserFactory):
    """Factory class for Dr3 regularisers
    """

    def create_with_action(
        self,
    ) -> Dr3WithAction:
        return Dr3WithAction()

    def create(
        self,
    ) -> Dr3:
        return Dr3()

    @staticmethod
    def get_type() -> str:
        return "dr3"


@dataclasses.dataclass()
class DefaultRegulariserFactory(RegulariserFactory):
    """Factory class for tjhe default regulariser i.e., no regularisation
    """

    def create_with_action(
        self,
    ) -> DefaultRegulariser:
        return DefaultRegulariser()

    def create(
        self,
    ) -> DefaultRegulariser:
        return DefaultRegulariser()

    @staticmethod
    def get_type() -> str:
        return "default"


(
    register_regulariser_factory,
    make_regulariser_field,
) = generate_config_registration(
    RegulariserFactory, lambda: DefaultRegulariserFactory()
)

register_regulariser_factory(DefaultRegulariserFactory)
register_regulariser_factory(Dr3RegulariserFactory)

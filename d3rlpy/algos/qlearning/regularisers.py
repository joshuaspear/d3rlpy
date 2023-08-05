from abc import ABCMeta, abstractmethod
import dataclasses
import torch
import torch.nn as nn
from typing import Any, Union, Tuple

from .base import QLearningAlgoImplBase
from ...torch_utility import TorchMiniBatch
from ...models.q_functions import Encoder, EncoderWithAction
from ...serializable_config import DynamicConfig, generate_config_registration


class Regulariser(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, algo:QLearningAlgoImplBase, batch:TorchMiniBatch
                 ) -> Tuple[float]:
        pass
    
class RegulariserPass(Regulariser):
    
    def __call__(self, algo:QLearningAlgoImplBase, batch:TorchMiniBatch
                 ) -> float:
        res = torch.tensor(0.0)
        return res

class _Dr3(Regulariser):
    """ DR3 Regularisation for approximate dynamic programming
    
    References:
        * `Kumar et al., DR3: VALUE-BASED DEEP REINFORCEMENT LEARNING REQUIRES 
        EXPLICIT REGULARIZATION. <https://openreview.net/pdf?id=POvMvLi91f>`_
    """
    # def __init__(self, encoder:Union[Encoder, EncoderWithAction]) -> None:
    #     self._prev_features = None
    #     self._encoder = encoder
    #     if isinstance(encoder, EncoderWithAction):
    #         self.get_feature_values = self._get_ewa_features_values
    #     elif isinstance(encoder, Encoder):
    #         self.get_feature_values = self._get_e_features_values
    #     else:
    #         raise ValueError("Incompatible encoder type")
            
        
    # def _get_e_features_values(self, x: torch.Tensor, action: torch.Tensor
    #                            ) -> torch.Tensor:
    #     return self._encoder(x=x)
    
    # def _get_ewa_features_values(self, x: torch.Tensor, action: torch.Tensor
    #                              ) -> torch.Tensor:
    #     return self._encoder(x=x, action=action)
    
    @abstractmethod
    def get_features(
        self, 
        encoder:Union[Encoder, EncoderWithAction], 
        observations:torch.Tensor, 
        action:torch.Tensor
        ):
        pass
    
    def __call__(self, algo:QLearningAlgoImplBase, batch:TorchMiniBatch
                 ) -> float:
        cur_features = []
        for q_func in algo.q_function._q_funcs:
            cur_features.append(
                self.get_features(
                    encoder=q_func.encoder, 
                    observations=batch.observations, 
                    action=batch.actions
                    )[None,:])
        next_features = []
        for q_func in algo.q_function._q_funcs:
            next_features.append(
                self.get_features(
                    encoder=q_func.encoder, 
                    observations=batch.next_observations, 
                    action=algo.predict_best_action(
                        x=batch.next_observations)
                    )[None,:])
        cur_features = torch.concat(cur_features).mean(dim=0)[:,None,:]
        next_features = torch.concat(next_features).mean(dim=0)[:,None,:]
        dot = torch.bmm(cur_features, next_features.transpose(1,2)).sum()
        return dot
                
        
class Dr3WithAction(_Dr3):
    
    def get_features(
        self, 
        encoder:EncoderWithAction,
        observations:torch.Tensor, 
        action:torch.Tensor
        ):
        return encoder(x=observations, action=action)
    
class Dr3(_Dr3):
    
    def get_features(
        self, 
        encoder:Encoder,
        observations:torch.Tensor, 
        action:torch.Tensor
        ):
        return encoder(x=observations)
    


@dataclasses.dataclass()
class RegulariserFactory(DynamicConfig):
    """
    """

    def create_with_action(
        self,
    ) -> Any:
        raise NotImplementedError()

    def create(
        self,
    ) -> Any:
        raise NotImplementedError()


@dataclasses.dataclass()
class Dr3RegulariserFactor(DynamicConfig):
    """
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
class RegulariserPassFactory(DynamicConfig):
    """
    """

    def create_with_action(
        self,
    ) -> RegulariserPass:
        return RegulariserPass()

    def create(
        self,
    ) -> RegulariserPass:
        return RegulariserPass()

    @staticmethod
    def get_type() -> str:
        return "pass"


register_regulariser_factory, make_regulariser_field = generate_config_registration(
    RegulariserFactory, lambda: RegulariserPassFactory()
)
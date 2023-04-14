from .dqn_impl import DoubleDQNImpl
from ...torch_utility import TorchMiniBatch
import torch

class DiscreteCRQLImpl(DoubleDQNImpl):
    
    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
            discount_reg=batch.q_regs
        )

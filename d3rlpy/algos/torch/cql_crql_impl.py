from .cql_impl import DiscreteCQLImpl
from ...torch_utility import TorchMiniBatch
import torch

class DiscreteCQLCRQLImpl(DiscreteCQLImpl):
            
    def __compute_dqn_loss(
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
            q_regs=batch.q_regs
        )
        
    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.__compute_dqn_loss(batch, q_tpn)
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long()
        )
        return loss + self._alpha * conservative_loss, conservative_loss

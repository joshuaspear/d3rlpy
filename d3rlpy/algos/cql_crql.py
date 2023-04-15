from .cql import DiscreteCQL 
from typing import Optional, Sequence
from .torch.cql_crql_impl import DiscreteCQLCRQLImpl

class DiscreteCQLCRQL(DiscreteCQL):
    
    _impl: Optional[DiscreteCQLCRQLImpl]

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteCQLCRQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

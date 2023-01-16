from typing import Optional

import numpy as np
from typing_extensions import Protocol

from ..dataset import Observation
from ..preprocessing import ActionScaler, ObservationScaler, RewardScaler

__all__ = ["AlgoProtocol"]


class AlgoProtocol(Protocol):
    def predict(self, x: Observation) -> np.ndarray:
        ...

    def predict_value(self, x: Observation, action: np.ndarray) -> np.ndarray:
        ...

    def sample_action(self, x: Observation) -> np.ndarray:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...

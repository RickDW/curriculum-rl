import numpy as np
from ray.rllib.agents.trainer import Trainer


def weight_vector(trainer: Trainer) -> np.ndarray:
    """
    Take a trainer's default policy and return its weights as a 1D vector.
    """
    policy = trainer.get_policy()
    weights = policy.get_weights()

    # make sure the weights are vectorized in a fixed order
    ordered_weights: list[np.ndarray] = [value for key, value in 
            sorted(weights.items(), key=lambda w: w[0]).items()]
    weight_vec = np.concatenate([weights.flatten() for weights in ordered_weights])

    return weight_vec

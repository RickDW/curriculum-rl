import gym
from ray.rllib.agents.registry import get_trainer_class

from typing import Union
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.agents.trainer import Trainer


# TODO
class CurriculumEnv(gym.Env):
    def __init__(self, trainer: Union[str, Trainer], trainer_config: TrainerConfigDict):
        super().__init__()

        if isinstance(trainer, Trainer):
            self.trainer = trainer(trainer_config)
        
        elif isinstance(trainer, str):
            self.trainer = get_trainer_class(trainer)(trainer_config)

        else:
            raise ValueError("trainer must be a string or a Trainer class " +
                f"instead of {type(trainer)}")

    def reset():
        # (re)initialize the learning agent / it's policy, and return its
        # current state as an observation of the curriculum agent
        pass

    def step(action):
        # set up the next task environment for the learning agent based on the
        # provided action. Once a training iteration is done, return the results
        # and the state of the learning agent
        pass


class LearningEnv(gym.Env):
    def __init__(self):
        super().__init__()

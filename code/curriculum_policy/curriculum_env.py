import gym
from ray.rllib.agents.registry import get_trainer_class

from typing import Union, Optional
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.agents.trainer import Trainer


# TODO
class CurriculumEnv(gym.Env):
    def __init__(
            self,
            trainer: Union[str, Trainer],
            curriculum_config: dict,
            trainer_config: TrainerConfigDict):
        
        super().__init__()

        if isinstance(trainer, Trainer):
            self.trainer = trainer
        elif isinstance(trainer, str):
            self.trainer = get_trainer_class(trainer)
        else:
            raise ValueError("trainer must be a string or a Trainer class " +
                f"instead of {type(trainer)}")

        self.trainer_instance: Optional[Trainer] = None
        self.trainer_config = trainer_config
        # TODO: the curriculum config should cover a number of things:
        #   - the state representation of the learning agent
        #   - the curriculum agent's rewards
        #   - the curriculum action space / the set of tasks that can be chosen
        self.curriculum_config = curriculum_config

    def init_trainer(self):
        self.trainer_instance = self.trainer(self.trainer_config)

    def reset(self):
        # (re)initialize the learning agent / it's policy, and return its
        # current state as an observation of the curriculum agent
        self.init_trainer()

    def step(action):
        # set up the next task environment for the learning agent based on the
        # provided action. Once a training iteration is done, return the results
        # and the state of the learning agent
        pass


# TODO
class LearningEnv(gym.Env):
    def __init__(self):
        super().__init__()

import gym

from curriculum_state import weight_vector
from curriculum_rewards import training_duration

from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.registry import get_trainer_class

from ray.rllib.utils.typing import TrainerConfigDict, ResultDict
from typing import Any, Union, Optional, Callable


class CurriculumEnv(gym.Env):
    """
    Allows a curriculum agent to interact with its learning agent.

    This environment additionally handles the training process of the learning
    agent, which runs inside the environment.
    """

    def __init__(
            self,
            trainer: Union[str, Trainer],
            trainer_config: TrainerConfigDict,
            target_task: Any,
            state_representation: str = "policy_weights",
            state_preprocessor: Callable[[Trainer], Any] = weight_vector,
            curriculum_rewards: Callable[[ResultDict], int] = training_duration):
        
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
        self.target_task = target_task

        self.state_representation = state_representation
        self.state_preprocessor = state_preprocessor
        self.curriculum_rewards = curriculum_rewards

    def init_trainer(self):
        self.trainer_instance = self.trainer(self.trainer_config)

    def get_state(self):
        """
        Returns the current state of the learning agent. Includes preprocessing
        """
        # TODO: implement other state representations?
        if self.state_representation == "policy_weights":
            return self.state_preprocessor(self.trainer_instance)
        else:
            raise ValueError(f"{self.state_representation} is not a valid "
                "curriculum state representation")

    def reset(self):
        # TODO shut down a possible previous trainer?
        self.init_trainer()
        return self.get_state()

    def step(self, action):
        #  TODO set up the next task environment for the learning agent based on the
        # provided action. Once a training iteration is done, return the results
        # and the state of the learning agent
        results = self.trainer_instance.train()

        state = self.get_state()
        reward = self.curriculum_rewards(results)
        # TODO: implement done check 
        done = False
        info = {}

        return state, reward, done, info


# TODO
class LearningEnv(gym.Env):
    def __init__(self):
        super().__init__()

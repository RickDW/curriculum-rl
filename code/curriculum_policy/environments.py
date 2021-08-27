import gym
import numpy as np

from state import weight_vector
from rewards import training_duration

from ray.rllib.agents.registry import get_trainer_class

from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.typing import TrainerConfigDict, ResultDict
from typing import Tuple, Any, Union, Optional, Callable, Type


class LearningEnv(gym.Env):
    """
    Abstract environment class that is used together with CurriculumEnv.

    # TODO: add documentation for curriculum_action_space, update()
    """

    def __init__(self):
        super().__init__()

    def update(self, action):
        raise NotImplementedError


def construct_default_spaces(trainer: Trainer, env_cls: Type[LearningEnv]) \
        -> Tuple[gym.Space, gym.Space, gym.Space]:
    """
    Define the default observation/action/reward spaces for CurriculumEnv.

    This method assumes the curriculum state is represented as a vector. The
    curriculum reward space is the unbounded interval (-inf, inf). The
    curriculum action space is determined by the specific LearningEnv that is
    used: its class is env_cls.
    """

    obs = trainer.get_state()
    assert isinstance(obs, np.ndarray)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)

    action_space = env_cls.curriculum_action_space

    reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    return observation_space, action_space, reward_space


class CurriculumEnv(gym.Env):
    """
    Allows a curriculum agent to interact with its learning agent.

    This environment additionally handles the training process of the learning
    agent, which is fully encapsulated by the CurriculumEnv.
    """

    def __init__(
            self,
            trainer: Union[str, Trainer],
            trainer_config: TrainerConfigDict,
            target_task: Any,
            state_representation: str = "policy_weights",
            state_preprocessor: Callable[[Trainer], Any] = weight_vector,
            curriculum_rewards: Callable[[ResultDict], int] = training_duration,
            obs_action_reward_definition: \
                Callable[[LearningEnv, Trainer], Tuple[gym.Space, gym.Space, gym.Space]] = \
                construct_default_spaces
            ):
        
        super().__init__()

        if isinstance(trainer, Trainer):
            self.trainer_cls = trainer
        elif isinstance(trainer, str):
            self.trainer_cls = get_trainer_class(trainer)
        else:
            raise ValueError("trainer must be a string or a Trainer class " +
                f"instead of {type(trainer)}")

        self.trainer: Optional[Trainer] = None
        self.trainer_config = trainer_config
        self.target_task = target_task
        self.ignore_next_init = False

        self.state_representation = state_representation
        self.state_preprocessor = state_preprocessor
        self.curriculum_rewards = curriculum_rewards

        self.init_trainer(first_init=True)
        env_cls = trainer_config["env"]
        # TODO remove temporary workaround, use the tune registry instead of explicit class
        assert isinstance(env_cls, LearningEnv)
        self.observation_space, self.action_space, self.reward_space = \
            obs_action_reward_definition(self.trainer, env_cls)


    def reset(self):
        self.init_trainer()
        return self.get_state()

    def step(self, action):
        self.update_task(action)

        results = self.trainer.train()

        state = self.get_state()
        reward = self.curriculum_rewards(results)
        # TODO: implement done check 
        done = False
        info = {}

        return state, reward, done, info

    def init_trainer(self, first_init=False):
        if self.ignore_next_init:
            # a new trainer has already been initialized
            self.ignore_next_init = False
            return

        if first_init:
            # the next time init_trainer() is called, don't create a new trainer
            self.ignore_next_init = True
        
        # TODO shut down a possible previous trainer?
        self.trainer = self.trainer_cls(self.trainer_config)

    def get_state(self):
        """
        Return the current state of the learning agent. Includes preprocessing
        """
        # TODO: implement other state representations?
        if self.state_representation == "policy_weights":
            return self.state_preprocessor(self.trainer)
        else:
            raise ValueError(f"{self.state_representation} is not a valid "
                "curriculum state representation")

    def update_task(self, action):
        """
        Go through all learning environments and update them with the new task
        """
        self.trainer.workers.foreach_worker(
            lambda w: w.foreach_env(
                # env needs to be an instance of LearningEnv
                lambda env: env.update(action)
            )
        )

"""
Implement the most basic example of the curriculum learning API
"""

import ray
from ray import tune
from ray import rllib
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
import gym
import random


class CurriculumEnv(TaskSettableEnv):
    def __init__(self, config: EnvContext):
        self.task = 0
        self.wrapped_env = gym.make('CartPole-v0')
        self.observation_space = self.wrapped_env.observation_space
        self.action_space = self.wrapped_env.action_space

    def reset(self):
        return self.wrapped_env.reset()

    def step(self, action):
        return self.wrapped_env.step(action)

    def sample_tasks(self, n_tasks):
        """Sample n random tasks."""
        return [random.randint(1, 10) for _ in range(n_tasks)]

    def get_task(self):
        "Get the current task."
        return self.task

    def set_task(self, task):
        "Set the task for this env."
        self.task = 0


def curriculum_fn(train_results: dict, env: TaskSettableEnv, \
        env_context: EnvContext) -> TaskType:
    # determine which task is trained on next
    new_task = 0
    return new_task # dummy implementation


if __name__ == "__main__":
    ray.init()

    config = {
        "env": CurriculumEnv,
        "env_config": {
            # environment parameters
        },
        "num_workers": 6,
        "env_task_fn": curriculum_fn,
        "framework": "tf2",

        # logging settings
        # "evaluation_interval": 1, # evaluate the agent after every training iteration
        # "evaluation_num_episodes": 5
    }

    results = tune.run(
        rllib.agents.ppo.PPOTrainer,
        config=config,
        stop={
            "episode_reward_mean": 200, # maximum reward in the cartpole environment
            "training_iteration": 50 # failsafe in case the agent cannot achieve the maximum reward
        },
        callbacks=[TBXLoggerCallback()],
        local_dir='~/Museum/mllab/curriculum-rl/results',
        verbose=2
    )

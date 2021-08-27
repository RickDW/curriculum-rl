import gym
import ray
from ray import tune

import curriculum_policy.environments as envs


class GridWorldNav(envs.LearningEnv):
    curriculum_action_space = gym.spaces.Discrete(9)

    # TODO: implement this env based on Narvekar's paper


ray.init()

# the config for the learning agent's trainer
learning_config = {
    "env": GridWorldNav,
    "env_config": {}
}

# the config for the curriculum agent's trainer
curriculum_config = {
    "env": envs.CurriculumEnv,
    "env_config": {
        "trainer": "PPO",
        "trainer_config": learning_config
    },
    "num_gpus": 0,
    "num_workers": 1
}

stop = {
    "training_iteration": 5
}

tune.run("PPO", name="curriculum_policy_test", local_dir="ray_results",
    config=curriculum_config, stop=stop)

ray.shutdown()

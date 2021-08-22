import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env

class CartpoleTest(MultiAgentEnv):
    def __init__(self):
        self.envs = [gym.make('CartPole-v0') for i in range(2)]

    def reset(self):
        obs_dict = {}

        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs_dict[f"agent_{i}"] = obs

        return obs_dict

    def step(self, action_dict):
        pass

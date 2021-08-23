"""
This code tests whether it is possible for an agent in a MultiAgentEnv to go
through multiple episodes while the MultiAgentEnv is not reset in between.
"""

import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env


class CartPoleTest(MultiAgentEnv):
    # agent IDs
    agent1 = "ag1"
    agent2 = "ag2"

    def __init__(self):
        self.counter = 0

        self.envs = {
            agentID: gym.make('CartPole-v0')
            for agentID in [self.agent1, self.agent2]
        }

        self.dones = {agentID: False for agentID in [self.agent1, self.agent2]}

    def reset(self):
        obs = {}

        for agentID, env in self.envs.items():
            obs[agentID] = env.reset()

        self.dones = {agentID: False for agentID in [self.agent1, self.agent2]}
        self.counter = 0

        return obs

    def step(self, action_dict):
        obs = {}
        reward = {}
        done = {}
        info = {}

        self.counter += 1

        for agentID, action in action_dict.items():
            env = self.envs.get(agentID, None)

            if not env:
                raise ValueError(f"Agent ID {agentID} is not recognized.")

            obs[agentID], reward[agentID], done[agentID], info[agentID] = \
                env.step(action)

            # keep track of which agents are done
            if done[agentID]:
                self.dones[agentID] = True

            # terminate agent one's episode after five steps
            if agentID == self.agent1 and self.counter == 5:
                assert not self.dones[agentID] # might not hold when policies have just been initialized
                # stop agent one's episode
                self.dones[agentID] = done[agentID] = True

        # start a new episode for agent one after ten steps
        if self.counter == 10:
            # agent one should not be active
            assert self.agent1 not in action_dict

            # start a new 'sub-episode' for agent one
            obs[self.agent1] = self.envs[self.agent1].reset()
            self.dones[self.agent1] = False

        # check if all agents are done
        done["__all__"] = all(self.dones.values())
            
        return obs, reward, done, info


register_env("CartPoleTest", lambda config: CartPoleTest())


dummy_env = gym.make("CartPole-v0")
obs_space = dummy_env.observation_space
action_space = dummy_env.action_space
ray.init()

tune.run(
    "PPO",
    name="sub_episode_env",
    local_dir="ray_results",
    config={
        "env": "CartPoleTest",
        "multiagent": {
            "policies": {
                agent_id: (None, obs_space, action_space, {})
                for agent_id in [CartPoleTest.agent1, CartPoleTest.agent2]
            },
            "policy_mapping_fn": (
                lambda agent_id, episode, **kwargs: agent_id),
        },
        "num_gpus": 0,
        "num_workers": 1
    },
    stop={
        "training_iteration": 10,
        "episode_reward_mean": 200
    }
)

ray.shutdown()
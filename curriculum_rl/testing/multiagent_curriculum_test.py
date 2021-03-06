import curriculum_policy.curriculum_env_old as cenv

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# TODO: Ignore the current design that's based on multi-agent setups and
#       nest a Trainer inside a single-agent curriculum environment.
#       The multi-agent approach could potentially work by using multiple agent
#       id's to fix the 'revival' issue, but there would still be the issue of
#       having to create custom trainers as soon as the curriculum agent and
#       the learning agent's training algorithms are not the same. This makes
#       the entire setup a lot less user-friendly and needlessly time-consuming
#       since you'd have to implement algorithms that are already provided by
#       RLlib. Even if this turns out to be relatively simple, it requires a
#       lot of advanced RLlib features which would probably cause a lot of 
#       researchers to ditch it and try something else.
#
#       My main concerns about this new approach are about the integration with
#       tune which should cover at least the basics such as logging,
#       hyperparameter sweeps, and experiment resource management.

# TODO: Implement curriculum policy learning stepwise to prevent wasting time
# * implement the next_task() functionality through callbacks
# * find a compact representation of the learning agent's state, e.g. tile
#       coding (refer to Sutton & Barto, Stone?)
# * do a test run with a standard trainer, i.e. do not risk implementing a
#       custom trainer for nothing. use the simplest possible env to speed things up
# * implement additional knowledge transfer methods (e.g. reward shaping) 
# * implement and test a custom trainer algorithm that mixes two algorithms
#       as a proof of concept
# * ONLY if this code turns out to be useful to someone else, consider using
#       PettingZoo as the basis for the multi-agent environment class. This
#       *might* make it possible to be used with libraries other than RLlib.
#       Additionally, RLlib should work fine with PettingZoo envs, so no
#       functionality should be lost

# TODO: standardize the config setup?

if __name__ == "__main__":
    ray.init()

    cenv.register_rllib()

    env_name = "CartPole-v0"
    env_config = {}
    obs_space, action_space = cenv.get_env_spaces(env_name)

    agent_ids = {
        "learning_agent": "LA", 
        "curriculum_agent": "AC"
    }


    tune.run(
        PPOTrainer,
        config={
            "env": "CurriculumWrapper",
            "env_config": {
                "env_name": env_name,
                "env_config": env_config,
                "agentIDs": agent_ids
            },
            "multiagent": {
                "policies": {
                    agent_ids["learning_agent"]: (
                        # (policy_cls, ..., ..., policy_config)
                        None, obs_space, action_space, {}
                        # None means default policy class, defined in trainer
                    ),
                    agent_ids["curriculum_agent"]: (
                        None, obs_space, action_space, {}
                    )
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: agent_id
            }
        },
        # TODO: implement the callbacks (train result, episode start)
        callbacks=[cenv.CurriculumCallback()],
        stop={
            "mean_episode_reward": 200,
            "training_iteration": 20
        }
    )

    ray.shutdown()

import gym
import ray
from ray import tune


if __name__ == "__main__":
    ray.init()

    tune.run(
        "PPO",
        name="rllib_boilerplate",
        local_dir="ray_results",
        config={
            "env": "CartPole-v0",
            "num_gpus": 0,
            "num_workers": 1
        },
        stop={
            "training_iteration": 1,
            "episode_reward_mean": 200
        }
    )

    ray.shutdown()

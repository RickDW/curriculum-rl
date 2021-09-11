import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.callbacks import DefaultCallbacks
from wandb import Video

from typing import Dict, Optional
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID

# Use this wrapper to use video recording 
# import gym
# gym.wrappers.Monitor


class VideoCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, 
            policies: Dict[PolicyID, Policy], episode: MultiAgentEpisode, 
            env_index: Optional[int], **kwargs) -> None:
        """
        Create a wandb Video object if a video has been registered in the info dict.
        """
            
        if not worker.multiagent:
            info = episode.last_info_for()
            if "video" in info:
                # there is a video available
                level = info["video"]["level"]
                path = info["video"]["path"]
                print(f"Creating video with path {path}")
                episode.media[f"level_{level}"] = Video(path)
        
        else:
            # handle multi-agent video recording
            raise NotImplementedError


ray.init()

analysis = tune.run(
    "PPO",
    config={
        "env": "CartPole-v0",

        "num_workers": 1,
        "num_envs_per_worker": 1,
        "record_env": "videos"

        # "evaluation_interval": 1, # evaluate every training iteration
        # "evaluation_num_episodes": 1, # stop evaluation after one episode
        # "evaluation_config": {
        #     "record_env": "videos",
        #     "callbacks": VideoCallback
        # }
    },
    callbacks=[
        WandbLoggerCallback(
            project="rllib_test",
            api_key_file="/workspaces/curriculum-rl/api_key.txt",
            monitor_gym=True # this calls wandb.gym.monitor() which modifies gym's Monitor wrapper
        )
    ],
    stop={
        "training_iteration": 2
    },
    local_dir="ray_results",
    verbose=1
)

ray.shutdown()

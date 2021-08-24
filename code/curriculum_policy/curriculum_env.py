import gym

from ray.tune.trainable import Trainable
from ray.tune.utils.placement_groups import PlacementGroupFactory

from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.utils.annotations import override

from ray.rllib.utils.typing import TrainerConfigDict
from typing import Union, Optional


class OverrideDefaultResourceRequest:
    @classmethod
    @override(Trainable)
    def default_resource_request(cls, config):
        print("hi there!")
        # cf = dict(cls._default_config, **config)

        # eval_config = cf["evaluation_config"]

        # # Return PlacementGroupFactory containing all needed resources
        # # (already properly defined as device bundles).
        # return PlacementGroupFactory(
        #     bundles=[{
        #         # Local worker + replay buffer actors.
        #         # Force replay buffers to be on same node to maximize
        #         # data bandwidth between buffers and the learner (driver).
        #         # Replay buffer actors each contain one shard of the total
        #         # replay buffer and use 1 CPU each.
        #         "CPU": cf["num_cpus_for_driver"] +
        #         cf["optimizer"]["num_replay_buffer_shards"],
        #         "GPU": cf["num_gpus"]
        #     }] + [
        #         {
        #             # RolloutWorkers.
        #             "CPU": cf["num_cpus_per_worker"],
        #             "GPU": cf["num_gpus_per_worker"],
        #         } for _ in range(cf["num_workers"])
        #     ] + ([
        #         {
        #             # Evaluation workers.
        #             # Note: The local eval worker is located on the driver CPU.
        #             "CPU": eval_config.get("num_cpus_per_worker",
        #                                    cf["num_cpus_per_worker"]),
        #             "GPU": eval_config.get("num_gpus_per_worker",
        #                                    cf["num_gpus_per_worker"]),
        #         } for _ in range(cf["evaluation_num_workers"])
        #     ] if cf["evaluation_interval"] else []),
        #     strategy=config.get("placement_strategy", "PACK"))


# add a mixin to a standard trainer
class PPOCustomResources(OverrideDefaultResourceRequest, PPOTrainer):
    pass


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

from ray.tune.trainable import Trainable
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.annotations import override


class OverrideDefaultResourceRequest:
    @classmethod
    @override(Trainable)
    def default_resource_request(cls, config):
        # TODO replace this dummy function
        cf = dict(cls._default_config, **config)

        eval_config = cf["evaluation_config"]

        # Return PlacementGroupFactory containing all needed resources
        # (already properly defined as device bundles).
        return PlacementGroupFactory(
            bundles=[{
                # Local worker + replay buffer actors.
                # Force replay buffers to be on same node to maximize
                # data bandwidth between buffers and the learner (driver).
                # Replay buffer actors each contain one shard of the total
                # replay buffer and use 1 CPU each.
                "CPU": cf["num_cpus_for_driver"] +
                cf["optimizer"]["num_replay_buffer_shards"],
                "GPU": cf["num_gpus"]
            }] + [
                {
                    # RolloutWorkers.
                    "CPU": cf["num_cpus_per_worker"],
                    "GPU": cf["num_gpus_per_worker"],
                } for _ in range(cf["num_workers"])
            ] + ([
                {
                    # Evaluation workers.
                    # Note: The local eval worker is located on the driver CPU.
                    "CPU": eval_config.get("num_cpus_per_worker",
                                           cf["num_cpus_per_worker"]),
                    "GPU": eval_config.get("num_gpus_per_worker",
                                           cf["num_gpus_per_worker"]),
                } for _ in range(cf["evaluation_num_workers"])
            ] if cf["evaluation_interval"] else []),
            strategy=config.get("placement_strategy", "PACK"))


# add a mixin to a standard trainer
class PPOCustomResources(OverrideDefaultResourceRequest, PPOTrainer):
    pass


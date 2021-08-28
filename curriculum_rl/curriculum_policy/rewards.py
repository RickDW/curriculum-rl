from ray.rllib.utils.typing import ResultDict


def agent_timesteps_total(training_results: ResultDict) -> int:
    """
    Calculate a curriculum reward based on how long it took to learn the task
    """
    return training_results["agent_timesteps_total"]

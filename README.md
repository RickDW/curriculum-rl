# Curriculum Learning Implementations for RL

This repository contains implementations of curriculum learning in reinforcement learning. The code uses the [RLlib library](https://docs.ray.io/en/latest/rllib.html).

## Work in progress

Currently the algorithm described in Narvekar and Stone's 2018 paper ["Learning Curriculum Policies for Reinforcement Learning"](https://arxiv.org/abs/1812.00285) is being implemented. The idea behind this algorithm is to let a second RL agent decide which tasks are part of the curriculum. This curriculum determines which tasks the main RL agent is trained on, and it is aimed at increasing the overall sample efficiency of the agent.

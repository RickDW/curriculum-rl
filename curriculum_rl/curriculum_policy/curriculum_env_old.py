import gym
from typing import String, Boolean, Float, Tuple, Dict, Any, Optional

from ray.tune import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from ray.rllib.agents.callbacks import DefaultCallbacks

# TODO: check whether early-exit and re-entry is possible for multi-agent envs !!
# TODO: MultiAgentEnv is not a subclass of gym.Env, add base class?
# TODO: write unittests for this class?
# TODO: give this curriculum env a maximum number of tasks? afterwards the episode could end
# TODO: check whether the different amounts of env steps (CA vs LA) is going to be a problem
class CurriculumWrapper(MultiAgentEnv):
    """
    A wrapper environment that supports curriculum-policy learning in RLlib.

    This is an abstract class. Concrete subclasses need to implement
    process_state(), process_results(), and process_task().

    The learning agent (LA) and the curriculum agent (CA) are identified by
    their agent IDs. These IDs are used to distinguish the observations, 
    rewards, and actions of the LA from those of the CA.

    The multi-agent API on which this class is based allows agents to perform
    actions at arbitrary timesteps. In this curriculum environment, an episode
    will always need the CA to perform an action before the LA does. This
    action determines which task the LA will be trained on in the next
    interval. The second action and all actions that come after it are chosen
    by the LA, and these actions are simply passed on to the wrapped
    environment. This will continue until a signal is given that indicates a
    new task needs to be chosen by the CA: this is given by calling next_task()

    The observations of the CA are based on the state of the LA, i.e. its
    policy, value function, or another model of interest. The CurriculumWrapper
    environment cannot access the state of the LA directly, and thus relies on
    the training algorithm for obtaining this state whenever needed. In RLlib,
    this is supported through callbacks which are provided in
    curriculum_callbacks.py. 
    
    The rewards of the CA are based on how many timesteps the LA used in the
    task that was specified by the CA. This information should be passed
    on to this environment just like the CA observations.
    """

    def __init__(self, env_name: String, agent_ids: Dict[String, AgentID],
            env_config: Optional[Dict]):
        """
        Prepare the learning agent's environment.
        """

        # TODO: use tune's/gym's registry here?
        self.env_name = env_name
        self.env_config = env_config
        self.env = gym.make(env_name, **env_config)

        # has next_task() been called?
        self.CA_feedback_available = False
        # are we waiting for the CA to choose a task?
        self.awaiting_task = False
        # has the CA chosen a task (equals False if a new curriculum episode
        # has just started and nothing has happened yet)
        self.first_task_chosen = False

        self.CA_observation = None
        self.CA_reward = None
        self.CA_done = False

        self.LA = agent_ids["learning_agent"]
        self.CA = agent_ids["curriculum_agent"]

        # TODO: specify action/reward/obs spaces


    def reset(self) -> MultiAgentDict:
        """
        Start a new episode for the learning agent and the curriculum agent.
        """

        if not self.CA_feedback_available:
            raise RuntimeError("The curriculum agent's observation and " +
                "reward are unknown (because next_task() was not called). " +
                "This is required for a call to reset() since it returns ")

        obs = {self.CA: self.CA_observation}

        self.awaiting_task = True
        self.first_task_chosen = False
        self.CA_feedback_available = False # don't use feedback more than once

        # reset all CA feedback (not required, but might help with debugging)
        self.CA_done = False
        self.CA_reward = None
        self.CA_observation = None

        return obs


    def next_task(self, LA_state: Any,
            train_results: Optional[Any] = None) -> None:
        """
        Put in a request for changing the learning agent's task.

        The train_results argument is optional when this function is called
        right after the learning agent has been initialized. In this case there
        are no training results available, and only the curriculum agent's
        observation is needed.
        """

        self.CA_observation = self.process_state(LA_state)
        if train_results:
            self.CA_reward, self.CA_done = self.process_results(train_results)

        self.CA_feedback_available = True


    def process_state(self, LA_state: Any) -> Any:
        """
        Process the state of the LA to obtain the observation of the CA.

        This function needs to be implemented in a subclass.
        """
        raise NotImplementedError
        # return LA_state


    def process_results(self, train_results: Any) -> Tuple[Float, Boolean]:
        """
        Determine the CA's reward and whether the curriculum episode is done.

        This function needs to be implemented in a subclass.
        """
        return 0.0, False

    def process_task(self, action: Any) -> None:
        """
        Set up a new task according to the curriculum agent's action.

        This function needs to be implemented in a subclass.
        """
        raise NotImplementedError


    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict,
            MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Go forward one timestep.
        """
        # add done["__all__"] to all return statements

        if len(action_dict) != 1:
            raise ValueError("At every timestep only one agent is allowed to" +
                " perform an action. action_dict has keys  " +
                f"{list(action_dict.keys())}")

        agent = next(iter(action_dict.keys()))
        action = action_dict[agent]
        
        #### Process a learning agent's action ################################
        if agent == self.LA:
            if not self.first_task_chosen and not self.awaiting_task:
                raise RuntimeError("The learning agent cannot do anything" +
                    " before the curriculum agent has chosen a(nother) task.")

            if self.CA_feedback_available:
                # a new task needs to be selected and the wrapped env is no
                # longer used
                # TODO: the learning agent's action is discarded here. The next
                # time the learning agent receives feedback it might relate it
                # to the discarded action. Since a new task is chosen however,
                # the wrapped env is discarded, and the learning agent's
                # episode should be reset. -> Check whether this reset actually
                # happens!
                # TODO: tear down wrapped env?
                obs = {self.CA: self.CA_observation}
                reward = {self.CA: self.CA_reward}
                done = {self.CA: self.CA_done} # TODO: read up on the 'done'
                # mechanism in multi-agent envs, check this implementation
                info = {self.CA: {}}

                self.CA_feedback_available = False # don't use feedback > once
                # remove CA feedback (not required but might help with debugging)
                self.CA_done = False
                self.CA_reward = None
                self.CA_observation = None

                return obs, reward, done, info

            else:
                # pass the learning agent's action on to its environment
                obs, reward, done, info = self.env.step(action)

                obs = {self.LA: obs}
                reward = {self.LA: reward}
                done = {self.LA: done}
                info = {self.LA: info}

                return obs, reward, done, info
                
        #### Process a curriculum agent's action ##############################
        elif agent == self.CA:
            if not self.awaiting_task:
                raise ValueError("The curriculum agent cannot chose a task" +
                    " right now.")

            # TODO: copy the 'if self.CA_feedback_available' block here for completion?

            self.process_task(action)
            
            self.awaiting_task = False # reset this flag
            self.first_task_chosen = True # in case this flag wasn't set yet
            

        #### Process an unknown agent's action ################################
        else:
            raise ValueError(f"Agent ID {agent} is not recognized.")


class CurriculumCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        return super().on_train_result(trainer, result, **kwargs)


def register_rllib():
    """
    Register the CurriculumWrapper env in the tune/rllib registry.
    """

    register_env(
        "CurriculumWrapper", 
        lambda env_context: CurriculumWrapper(**env_context)
    )


def get_env_spaces(env_name, env_config = {}):
    """
    Instantiate an environment and return its observation and action spaces.
    """
    # TODO: use the tune registry / allow users to choose?
    env = gym.make(env_name, **env_config)

    obs_space = env.observation_space
    action_space = env.action_space

    return obs_space, action_space

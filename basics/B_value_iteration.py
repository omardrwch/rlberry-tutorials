import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rlberry.agents import AgentWithSimplePolicy
from rlberry.envs.finite import FiniteMDP
from rlberry.envs import GridWorld
from rlberry import types


class ValueIterationAgent(AgentWithSimplePolicy):
    """
    Agent that performs value iteration for a finite MDP.

    Parameters
    ----------
    env: types.Env
        Environment.
    gamma: float
        Discount factor.
    **kwargs: dict
        Arguments for base class.
    """

    name = "VI"

    def __init__(self, env: types.Env, gamma: float = 0.99, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        env = self.env
        # discount factor
        self.gamma = gamma
        # check the env is a FiniteMDP, which gives access to
        # the reward function and transition probabilities
        assert isinstance(env, FiniteMDP)
        # number of states and actions
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        # initialize value functions
        self.q_func = np.zeros((self.n_states, self.n_actions))
        self.v_func = np.zeros(self.n_states)
        # counter for number of iterations and current error
        self.n_iterations = 0
        self.error = np.inf

    def fit(self, budget, **kwargs):
        """
        Parameters
        ----------
        budget: int
            Number of iterations to perform
        """
        del kwargs
        for _ in range(budget):
            # Apply Bellman operator
            new_q = self.env.R + self.gamma * self.env.P @ self.v_func
            new_v = new_q.max(axis=-1)

            # Update variables
            self.error = np.abs(new_q - self.q_func).max()
            self.n_iterations += 1
            self.q_func = new_q
            self.v_func = new_v

            # Write useful information
            self._log_info()

    def _log_info(self):
        if self.writer:
            self.writer.add_scalar("error", self.error, self.n_iterations)
            # every 10 iterations, evaluate policy
            if self.n_iterations % 10 == 0:
                estimated_value = self.eval(
                    eval_horizon=1.0 / (1.0 - self.gamma),
                    n_simulations=10,
                    gamma=1.0,  # compute total reward for evaluation
                )
                self.writer.add_scalar(
                    "estimated_value", estimated_value, self.n_iterations
                )

            # every 100 iterations, save image representing
            # the value function in tensorboard
            if self.n_iterations % 100 == 0 and isinstance(self.env, GridWorld):
                value_img = self.env.get_layout_img(self.v_func)
                value_img_tensor = torch.tensor(value_img)
                self.writer.add_image(
                    "value",
                    value_img_tensor,
                    self.n_iterations,
                    dataformats="HWC",
                )
                

    def policy(self, observation):
        return self.q_func[observation, :].argmax()


if __name__ == "__main__":
    env_layout = """
IOOOO # OOOOO  O OOOOO
OOOOO # OOOOO  # OOOOO
OOOOO O OOOOO  # OOOOO
OOOOO # OOOOO  # OOOOO
IOOOO # OOOOO  # OOOOr
"""
    env = GridWorld.from_layout(layout=env_layout, success_probability=0.9)
    agent = ValueIterationAgent(env, gamma=0.99)
    agent.fit(300)
    print(agent.error)

    #
    # Visualize the learned policy
    #
    env.enable_rendering()
    state = env.reset()
    for tt in range(100):
        action = agent.policy(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            state = env.reset()
    env.render()

    # Visualize logged data
    writer_data = agent.writer.data
    error_data = writer_data[writer_data["tag"] == "error"]
    value_data = writer_data[writer_data["tag"] == "estimated_value"]

    plt.figure()
    sns.lineplot(data=error_data, x="global_step", y="value")
    plt.xlabel("number of iterations")
    plt.ylabel("error")

    plt.figure()
    sns.lineplot(data=value_data, x="global_step", y="value")
    plt.xlabel("number of iterations")
    plt.ylabel("estimated_value")

    plt.show()

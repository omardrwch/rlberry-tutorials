import numpy as np
from rlberry.agents import AgentWithSimplePolicy
from rlberry.envs.finite import FiniteMDP


class ValueIterationAgent(AgentWithSimplePolicy):
    def __init__(self, env, gamma=0.99, **kwargs):
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
            self.writer.add_scalar('error', self.error, self.n_iterations)

    def policy(self, observation):
        return self.q_func[observation, :].argmax()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rlberry.envs import GridWorld

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
    error_data = writer_data[writer_data['tag'] == 'error']
    sns.lineplot(data=error_data, x="global_step", y="value")
    plt.xlabel("number of iterations")
    plt.ylabel("error")
    plt.show()

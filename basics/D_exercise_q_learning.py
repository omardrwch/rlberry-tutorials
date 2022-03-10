from rlberry.agents import AgentWithSimplePolicy


class QLearningAgent(AgentWithSimplePolicy):
    name = "Q-Learning"

    # you can include more parameters in the constructor!
    def __init__(self, env, gamma=0.99, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        env = self.env
        self.gamma = gamma

    def fit(self, budget, **kwargs):
        # To be implemented!
        pass

    def policy(self, observation):
        # To be implemented!
        return self.env.action_space.sample()


if __name__ == "__main__":
    # To be implemented!
    # Compare your Q-learning implementation to value iteration.
    # You can take inspiration from the script C_evaluate_and_compare.py
    pass

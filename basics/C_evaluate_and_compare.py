from B_value_iteration import ValueIterationAgent
from rlberry import types
from rlberry.envs import GridWorld
from rlberry.manager import AgentManager
from rlberry.manager import evaluate_agents
from rlberry.manager import plot_writer_data


class SampledValueIterationAgent(ValueIterationAgent):
    """
    Alternative implementation of value iteration that,
    at each iteration, applies the Bellman operator only for a single state,
    that is uniformly sampled from all possible states.

    Parameters
    ----------
    env: types.Env
        Environment.
    gamma: float
        Discount factor.
    **kwargs: dict
        Arguments for base class.
    """

    name = "Sampled VI"

    def __init__(self, env: types.Env, gamma: float = 0.99, **kwargs):
        ValueIterationAgent.__init__(self, env, gamma, **kwargs)

    def fit(self, budget, **kwargs):
        """
        Parameters
        ----------
        budget: int
            Number of iterations to perform
        """
        del kwargs
        for _ in range(budget):
            # Apply Bellman operator to a sampled state
            state = self.env.observation_space.sample()
            self.q_func[state] = (
                self.env.R[state] + self.gamma * self.env.P[state] @ self.v_func
            )
            self.v_func[state] = self.q_func[state].max()

            # Update variables
            self.n_iterations += 1

            # Write useful information
            self._log_info()


def env_constructor():
    env_layout = """
IOOOO # OOOOO  O OOOOO
OOOOO # OOOOO  # OOOOO
OOOOO O OOOOO  # OOOOO
OOOOO # OOOOO  # OOOOO
IOOOO # OOOOO  # OOOOR
"""
    env = GridWorld.from_layout(layout=env_layout, success_probability=0.9)
    return env


if __name__ == "__main__":

    # Tuple (constructor, kwargs)
    env = (env_constructor, dict())

    # Number of iterations to run
    N_ITERATIONS = 5000

    # Agent parameters
    params = dict(gamma=0.99)

    # Evaluation paramgers
    eval_kwargs = dict(eval_horizon=100)

    # Agent manager for (usual) value iteration
    manager_vi = AgentManager(
        ValueIterationAgent,
        train_env=env,
        init_kwargs=params,
        eval_kwargs=eval_kwargs,
        fit_budget=N_ITERATIONS,
        n_fit=4,
        parallelization="process",
        output_dir="rlberry_data/value_iteration",
        seed=123,
        enable_tensorboard=True,
    )

    # Agent manager for "sampled" value iteration
    manager_sampled_vi = AgentManager(
        SampledValueIterationAgent,
        train_env=env,
        init_kwargs=params,
        eval_kwargs=eval_kwargs,
        fit_budget=N_ITERATIONS,
        n_fit=4,
        parallelization="process",
        output_dir="rlberry_data/sampled_value_iteration",
        seed=123,
        enable_tensorboard=True,
    )

    # Run agent managers
    manager_vi.fit()
    manager_sampled_vi.fit()

    # Visualize perfomance during learning
    all_managers = [manager_vi, manager_sampled_vi]
    plot_writer_data(all_managers, tag="estimated_value", show=False)
    plot_writer_data(all_managers, tag="dw_time_elapsed", show=False)

    # Evaluate performance of final policy
    evaluate_agents(all_managers, n_simulations=10, show=True)

    # To vistualize with tensorboard, run:
    # $ tensorboard --logdir rlberry_data

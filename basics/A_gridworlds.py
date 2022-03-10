from rlberry.envs import GridWorld


#
# Step 1: define GridWorld layout
#

"""
Layout symbols:

'#' : wall
'r' : reward of 1, terminal state
'R' : reward of 1, non-terminal state
'T' : terminal state
'I' : initial state (if several, start uniformly among I)
'O' : empty state
any other char : empty state
"""

env_layout = """
IOOOO # OOOOO  O OOOOR
OOOOO # OOOOO  # OOOOO
OOOOO O OOOOO  # OOOOO
OOOOO # OOOOO  # OOOOO
IOOOO # OOOOO  # OOOOr
"""

if __name__ == "__main__":
    #
    # Step 2: create GridWorld instance
    #
    env = GridWorld.from_layout(layout=env_layout, success_probability=0.9)

    #
    # Access reward function and transition probabilities
    #
    state = env.observation_space.sample()
    action = env.action_space.sample()
    print(f"(state, action) = (s, a) = ({state}, {action})")
    print(f"expected reward at (state, action) = {env.R[state, action]}")
    print(f"transitions P[s'|s, a] for all s' = {env.P[state, action, :]}")

    #
    # Visualize a random policy
    #
    env.enable_rendering()
    state = env.reset()
    for tt in range(100):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # ...
        state = next_state
        if done:
            state = env.reset()
    env.render()

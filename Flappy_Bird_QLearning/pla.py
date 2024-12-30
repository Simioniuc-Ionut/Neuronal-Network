import numpy as np
import flappy_bird_gymnasium
import gymnasium


# Configurare
env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True,)


state_bins = [
    np.linspace(-1, 1, 10),  # y_bird discretizat
    np.linspace(-1, 1, 5),  # v_y discretizat
    np.linspace(0, 1, 10),  # x_pipe discretizat
    np.linspace(-1, 1, 5)  # y_pipe discretizat
]
num_states = (10, 5, 10, 5)  # Dimensiunile stărilor discretizate
num_actions = env.action_space.n

# Inițializare Q-table
Q = np.zeros(num_states + (num_actions,))



# Q-Learning
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
num_episodes = 1000

for episode in range(num_episodes):
    # state = discretize_state(env.reset())
    done = False
    # total_reward = 0
    state = env.reset()
    print("Prima stare dupa reset : ",state)
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing:
        state, reward, terminated, _, info = env.step(action)
        print("state : ",env.render().shape)
        # Checking if the player is still alive
        if terminated:
            break

    epsilon *= epsilon_decay
    # print(f"Episode {episode}: Total Reward: {total_reward}")

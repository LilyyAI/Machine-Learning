import numpy as np

states = 5
actions = 2

q_table = np.zeros((states, actions))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2
episodes = 100

for episode in range(episodes):
    state = 0
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(q_table[state])

        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(states - 1, state + 1)

        reward = 10 if next_state == states - 1 else -1
        done = next_state == states - 1

        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state

print("Q-Table:")
print(q_table)

state = 0
path = [state]

while state != states - 1:
    action = np.argmax(q_table[state])

    if action == 0:
        state = max(0, state - 1)
    else:
        state = min(states - 1, state + 1)

    path.append(state)

print("\nLearned Path:")
print(path)
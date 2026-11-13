
import numpy as np

states = 5
actions = 2
goal = 4

Q = np.zeros((states, actions))

learning_rate = 0.8
discount_factor = 0.9
epsilon = 0.2
episodes = 100

def step(state, action):
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(states - 1, state + 1)

    reward = 10 if next_state == goal else -1
    done = next_state == goal

    return next_state, reward, done

for episode in range(episodes):
    state = 0

    while state != goal:
        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Learned Q-Table")
print(np.round(Q, 2))

print("\nBest Actions")
for state in range(states):
    print("State", state, ":", "Left" if np.argmax(Q[state]) == 0 else "Right")

print("\nLearned Path")
state = 0
path = [state]

while state != goal:
    action = np.argmax(Q[state])
    state, reward, done = step(state, action)
    path.append(state)

print(path)
print("\nGoal Reached:", state == goal)

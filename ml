Deep Learning:
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
# Load top 10,000 words from IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# Pad sequences to the same length (200 words)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')  # Output: 0 (neg) or 1 (pos)
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

CNN:
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# Parameters
num_classes = 10
input_shape = (28, 28, 1)
# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Reshape to (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
# Train
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

Q Learning:
import numpy as np
# Q-learning settings
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000  # Training episodes
# Environment: 5x5 grid, goal at (4, 4)
grid_size = 5
num_actions = 4  # up, down, left, right
q_table = np.zeros((grid_size * grid_size, num_actions))
def state_to_index(state):
    return state[0] * grid_size + state[1]
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(q_table[state_to_index(state)])
def take_action(state, action):
    row, col = state
    if action == 0 and row > 0: row -= 1     # Up
    elif action == 1 and row < 4: row += 1   # Down
    elif action == 2 and col > 0: col -= 1   # Left
    elif action == 3 and col < 4: col += 1   # Right
    next_state = (row, col)
    reward = 1 if next_state == (4, 4) else -0.1
    done = next_state == (4, 4)
    return next_state, reward, done
# Q-learning main loop
for ep in range(episodes):
    state = (0, 0)
    while True:
        action = choose_action(state)
        next_state, reward, done = take_action(state, action)
        i, ni = state_to_index(state), state_to_index(next_state)
        q_table[i, action] += alpha * (reward + gamma * np.max(q_table[ni]) - q_table[i, action])
        state = next_state
        if done:
            break
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1} complete")
# Test learned policy
def test_policy():
    state = (0, 0)
    path = [state]
    while state != (4, 4):
        action = np.argmax(q_table[state_to_index(state)])
        state, _, _ = take_action(state, action)
        path.append(state)
    return path
print("Learned path to goal:", test_policy())

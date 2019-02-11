import numpy as np
import cv2
import os

def visualize_values(dqn, states, goal):
    values = dqn.get_values(states, np.array([goal]*len(states)))
    return np.reshape(values, [10, 10])

def visualize_all_values(dqn, states):
    all_goals = np.copy(states)
    for i, goal in enumerate(all_goals):
        values = visualize_values(dqn, states, goal)
        #values = 255*((values - np.min(values)) / (np.max(values) - np.min(values)))
        values = 255*values
        values = cv2.resize(values, (400, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        values = np.tile(np.reshape(values, (400, 400, 1)), (1,1,3))
        print(values.shape)
        print(cv2.imwrite(os.path.join('value_visualizations', f'{i}.png'), values))
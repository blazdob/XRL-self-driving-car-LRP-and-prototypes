import numpy as np

#driving forward:
def create_forward_prototype(length = 50, action = 2):
    actions = [action] * length
    # states with lenght vectors each of length 5
    states = np.zeros((length,5))
    for i, state in enumerate(states):
        # decreasing state 2
        state[0] = 0.2
        state[1] = 0.4
        state[2] = (length - i) / length
        state[3] = 0.4
        state[4] = 0.2
    return zip(actions, states)
    
#driving left:
def create_left_prototype(length = 50, action = 0):
    actions = [action] * length
    # states with lenght vectors each of length 5
    states = np.zeros((length,5))
    for i, state in enumerate(states):
        # decreasing state 2
        state[0] = 0.2
        state[1] = 0.4
        state[2] = 0.4
        state[3] = (length - i) / length
        state[4] = 0.2
    return zip(actions, states)

#driving right:
def create_right_prototype(length = 50, action = 1):
    actions = [action] * length
    # states with lenght vectors each of length 5
    states = np.zeros((length,5))
    for i, state in enumerate(states):
        # decreasing state 2
        state[0] = 0.2
        state[1] = (length - i) / length
        state[2] = 0.4
        state[3] = 0.4
        state[4] = 0.2
    return zip(actions, states)

import sys
# append relative path from /Users/blazdobravec/Documents/FACULTY/DOC/RW1/INITIAL_RESEARCH/RL_self_driving_car/self_driving_car_env
sys.path.append('/Users/blazdobravec/Documents/FACULTY/DOC/RW1/INITIAL_RESEARCH/RL_self_driving_car')
from self_driving_car_env.env import SelfDrivingCar
import pygame
import numpy as np

def normalize_state(state):
    """
    Normalizes the state space.
    """
    max_sensor = max(state[0] - min(state[0]))
    # set to absolute value
    state[0] = state[0] - min(state[0])
    # normalize numpy array state 
    normalized_state = state[0] / max_sensor
    return normalized_state


def record_game():
    game = SelfDrivingCar(render_mode=True, human_control=True)
    # numpy array of 5 sensors values 0
    state = game.reset()
    action = 2
    lst_of_as_tuples = []
    while True:
        # get action from keyboard input if held down
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 0
                else:
                    action = 2
            else:
                action = 2
        lst_of_as_tuples.append((action, state))
        next_state, reward, done, info = game.step(action)
        state = next_state
        # print(next_state, reward, done, info)
        if done:
            game.reset()
            break
    return lst_of_as_tuples


if __name__ == "__main__":
    lst_of_as_tuples = record_game()
    # save to file
    import pickle
    with open('recorded_game.pkl', 'wb') as f:
        pickle.dump(lst_of_as_tuples, f)
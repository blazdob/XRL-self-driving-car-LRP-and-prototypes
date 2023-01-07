import numpy as np
import pickle
from os.path import join, exists
CSV_DELIMITER = ','

S_A_TABLE_FILE_NAME = 'seq-a-table_pads1_prototype.pickle'
S_S_TABLE_FILE_NAME = 'seq-s-table_pads1_prototype.pickle'


class PrototypeTracker(object):

    def __init__(self, num_episodes):
        self.s_a = [[]]
        """ Corresponds to the sequence of actions performed by the agent at each time-step. """

        self.s_s = [[]]
        """ Corresponds to the sequence of states visited/observed by the agent at each time-step. """

        self.num_episodes = num_episodes
        self._cur_episode = 0

        left_prototype_path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/INITIAL_RESEARCH/RL_self_driving_car/prototypes/defined_prototypes/left_turning.npy"
        right_prototype_path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/INITIAL_RESEARCH/RL_self_driving_car/prototypes/defined_prototypes/right_turning.npy"
        straight_prototype_path = "/Users/blazdobravec/Documents/FACULTY/DOC/RW1/INITIAL_RESEARCH/RL_self_driving_car/prototypes/defined_prototypes/straight.npy"
        self.left_prototype = np.load(left_prototype_path)
        self.right_prototype = np.load(right_prototype_path)
        self.straight_prototype = np.load(straight_prototype_path)

        self.reset()

    def get_states(self):
        """
        Returns the sequence of states visited by the agent at each time-step.
        :return: the sequence of states visited by the agent at each time-step.
        """
        return self.s_s
    
    def get_actions(self):
        """
        Returns the sequence of actions performed by the agent at each time-step.
        :return: the sequence of actions performed by the agent at each time-step.
        """    
        return self.s_a
    
    def reset(self):
        """
        Resets the tracker by cleaning the state and action trajectories.
        :return:
        """
        self.s_a = [[] for _ in range(self.num_episodes)]
        self.s_s = [[] for _ in range(self.num_episodes)]
        self._cur_episode = 0

    def new_episode(self):
        """
        Signals the tracker that a new episode has started.
        :return:
        """
        if self._cur_episode < self.num_episodes - 1:
            self._cur_episode += 1

    def add_sample(self, state, action):
        """
        Adds a new state-action pair sample to the tracker
        :param int state: the visited state.
        :param int action: the executed action.
        :return:
        """
        self.s_s[self._cur_episode].append(state)
        self.s_a[self._cur_episode].append(action)
    
    def current_prototype_meta_action(self, n):
        """
        Defines the main possible multi-actions
        """


        #get last 25 states
        last_n_states = self.s_s[self._cur_episode][-n:]
        #normalize last 25 states using normalize_state_space function
        n_and_d_last_n_states = PrototypeTracker.discretize_state_space(PrototypeTracker.normalize_state_space(last_n_states), 3)
        # # compare it to the prototypes
        # # compare the same index of the last 25 states to the prototypes
        left_prototype_score = 0
        right_prototype_score = 0
        straight_prototype_score = 0
        for i in range(len(n_and_d_last_n_states)):
            # calculate error
            left_prototype_error = abs(n_and_d_last_n_states[i] - self.left_prototype[i])
            right_prototype_error = abs(n_and_d_last_n_states[i] - self.right_prototype[i])
            straight_prototype_error = abs(n_and_d_last_n_states[i] - self.straight_prototype[i])
            # add error to the score
            left_prototype_score += left_prototype_error
            right_prototype_score += right_prototype_error
            straight_prototype_score += straight_prototype_error
        # print("l_error: ", round(sum(left_prototype_error),2), "r_error: ", round(sum(right_prototype_error),2), "s_error: ", round(sum(straight_prototype_error),2))
        # return the action with the lowest error
        #print round and sumed up scores
        # print("l_score: ", round(sum(left_prototype_score),2), "r_score: ", round(sum(right_prototype_score),2), "s_score: ", round(sum(straight_prototype_score),2))
        sum_left_prototype_score = sum(left_prototype_score)
        sum_right_prototype_score = sum(right_prototype_score)
        sum_straight_prototype_score = sum(straight_prototype_score)
        if sum_left_prototype_score < sum_right_prototype_score and sum_left_prototype_score < sum_straight_prototype_score:
            return 1
        elif sum_right_prototype_score < sum_left_prototype_score and sum_right_prototype_score < sum_straight_prototype_score:
            return 0
        elif sum_straight_prototype_score < sum_left_prototype_score and sum_straight_prototype_score < sum_right_prototype_score:
            return 2


        # if it matches the best, return the action that is associated with that prototype
        # if it doesn't match any of the prototypes, return the action that is associated with the prototype that has the lowest error

            


    def save(self, output_dir):
        """
        Saves all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory in which to save the data.
        :return:
        """
        PrototypeTracker.write_table_pickle(self.s_a, join(output_dir, S_A_TABLE_FILE_NAME))
        PrototypeTracker.write_table_pickle(self.s_s, join(output_dir, S_S_TABLE_FILE_NAME))

    def load(self, output_dir):
        """
        Loads all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory from which to load the data.
        :return:
        """

        if not exists(output_dir):
            return
        self.s_a = PrototypeTracker.read_table_pickle(join(output_dir, S_A_TABLE_FILE_NAME))
        self.s_s = PrototypeTracker.read_table_pickle(join(output_dir, S_S_TABLE_FILE_NAME))
        self.num_episodes = self._cur_episode = len(self.s_a)

    @staticmethod
    def write_table_pickle(table, csv_file_path):
        """
        Writes the given array into a CSV file.
        :param array_like table: the data to be written to the CSV file.
        :param str csv_file_path: the path to the CSV file in which to write the data.
        :param str delimiter: the delimiter for the fields in the CSV file.
        :param str fmt: the string used to format the output of the elements in the data.
        :param array_like col_names: a list containing the names of each column of the data.
        :return:
        """
        #save to pickle
        with open(csv_file_path, 'wb') as f:
            pickle.dump(table, f)
        
    @staticmethod
    def read_table_pickle(csv_file_path):
        """
        Loads an array from a CSV file.
        :param str csv_file_path: the path to the CSV file from which to load the data.
        :param str delimiter: the delimiter for the fields in the CSV file.
        :param object dtype: the type of the elements stored in the data file.
        :param bool has_header: whether the first line of the CSV file contains the column names.
        :return np.ndarray: the numpy array loaded from the CSV file.
        """
        # return readed pickle file
        with open(csv_file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def normalize_state_space(states):
        """
        Normalizes the state space.
        """
        normalized_states = []
        for state in states:
            max_sensor = max(state[0] - min(state[0]))
            # set to absolute value
            state[0] = state[0] - min(state[0])
            # normalize numpy array state 
            normalized_state = state[0] / max_sensor
            normalized_states.append(normalized_state)
        return normalized_states

    @staticmethod 
    def discretize_state_space(all_states, precision):
        """
        Discretizes the state space with a given precision.
        """
        discretized_states = []
        for episode_states in all_states:
            episode_disc_states = np.round_(episode_states, decimals = precision, out = None)
            discretized_states.append(episode_disc_states)
        return discretized_states
import numpy as np
import pickle
from os.path import join, exists
CSV_DELIMITER = ','

S_A_TABLE_FILE_NAME = 'seq-a-table_pads1.pickle'
S_S_TABLE_FILE_NAME = 'seq-s-table_pads1.pickle'


class BehaviorTracker(object):

    def __init__(self, num_episodes):
        self.s_a = [[]]
        """ Corresponds to the sequence of actions performed by the agent at each time-step. """

        self.s_s = [[]]
        """ Corresponds to the sequence of states visited/observed by the agent at each time-step. """

        self.num_episodes = num_episodes
        self._cur_episode = 0

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

    def current_meta_action(self, rolling_window):
        """
        Defines the main possible multi-actions
        """
        # select last 20 actions and get the most occuring action
        try:
            last_rolling_window_actions = self.s_a[self._cur_episode][-rolling_window:]
            action = max(set(last_rolling_window_actions), key=last_rolling_window_actions.count)
            return action
        except:
            pass

    def save(self, output_dir):
        """
        Saves all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory in which to save the data.
        :return:
        """
        BehaviorTracker.write_table_pickle(self.s_a, join(output_dir, S_A_TABLE_FILE_NAME))
        BehaviorTracker.write_table_pickle(self.s_s, join(output_dir, S_S_TABLE_FILE_NAME))

    def load(self, output_dir):
        """
        Loads all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory from which to load the data.
        :return:
        """

        if not exists(output_dir):
            return
        self.s_a = BehaviorTracker.read_table_pickle(join(output_dir, S_A_TABLE_FILE_NAME))
        self.s_s = BehaviorTracker.read_table_pickle(join(output_dir, S_S_TABLE_FILE_NAME))
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


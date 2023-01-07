import numpy as np
import matplotlib.pyplot as plt
#import pca library
from sklearn.decomposition import PCA

class StateFrequencyAnalysis():
    """
    Represents an analysis of an agent's frequently-visited states, association patterns in the state features and
    association rules denoting consistent causality effects in the environment.
    """

    def __init__(self, states, nbins=1000):
        """
        Creates a new frequent states analysis.
        """
        self.states = [episode[1:] for episode in states]
        self.nbins = nbins

    def analyze(self):
        
        # discretizes the state space (with (5, ) vector as state representation)
        normalized_states = self.normalize_state_space()
        discretized_states = self.discretize_state_space(normalized_states, 1)
        # frequency distribution of states
        visited_states_counts = self.get_state_freq_dist(discretized_states)

        # performs PCA on the discretized states
        pca_flattened_states, pca_states = self.pca_states(self.states)
        
        # plots the PCA states
        self.plot_pca_states(pca_flattened_states)

        # perform kmeans clustering on the PCA states
        kmeans = self.kmeans_clustering(pca_flattened_states, 3)
        
        # plots the kmeans clustering
        self.plot_kmeans_clustering(pca_flattened_states, kmeans)


    def discretize_state(self, state, precision):
        """
        Discretizes the state space with a given precision.
        """
        
        
    def normalize_state_space(self):
        """
        Normalizes the state space.
        """
        normalized_states = []
        for episode_states in self.states:
            normalized_episode_states = []
            for state in episode_states:
                max_sensor = max(state[0] - min(state[0]))
                # set to absolute value
                state[0] = state[0] - min(state[0])
                # normalize numpy array state 
                normalized_state = state[0] / max_sensor
                normalized_episode_states.append(normalized_state)
            normalized_states.append(normalized_episode_states)
        return normalized_states

    def discretize_state_space(self, all_states, precision):
        """
        Discretizes the state space with a given precision.
        """
        discretized_states = []
        for episode_states in all_states:
            episode_disc_states = np.round_(episode_states, decimals = precision, out = None)
            discretized_states.append(episode_disc_states)
        return discretized_states
    

    def get_state_freq_dist(self, discretized_states):
        """
        Gets the state frequency distribution.
        """
        state_freq_dct = {}
        for episode_states in discretized_states:
            for state in episode_states:
                if state == []:
                    continue
                tup = tuple(state)
                state_freq_dct[tup] = state_freq_dct.get(tup, 0) + 1
        return state_freq_dct

    
    def plot_state_freq_dist(self, state_freq_dct):
        """
        Plots the state frequency distribution.
        """
        plt.bar([str(state) for state in state_freq_dct.keys()], state_freq_dct.values(), color='g')
        plt.show()
    
    def pca_states(self, states):
        """
        Performs PCA on the states.
        """
        pca = PCA(n_components=1)
        # flatten states
        flattened_states = []
        for episode_states in states:
            for state in episode_states:
                flattened_states.append(state[0])
        # states = np.asarray(states).flatten().flatten()
        pca.fit(flattened_states)
        pca_flattened_states = pca.transform(flattened_states)
        # transform back to the shape that was before
        pca_states = []
        for i in range(1, len(states)):
            episode_states = []
            for j in range(len(states[i])):
                try:
                    episode_states.append(pca_flattened_states[i * (len(states[i])) + j])
                except:
                    break
            pca_states.append(episode_states)
        return pca_flattened_states, pca_states
    
    def plot_pca_states(self, pca_flattened_states):
        """
        Plots the PCA states.
        """
        plt.plot(pca_flattened_states)
        plt.show()
    
    def kmeans_clustering(self, pca_states, n_clusters):
        """
        Performs kmeans clustering on the PCA states.
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters)
        kmeans.fit(pca_states)
        print(kmeans.labels_)
        return kmeans.labels_

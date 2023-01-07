
import matplotlib.pyplot as plt

class ActionSequenceAnalysis(object):
    """
    Analyzes the sequence of actions performed by the agent.
    """

    def __init__(self, episode_actions):
        """
        :param states: the sequence of states visited by the agent.
        """
        self.episode_actions = episode_actions
        # remove empyt lists
        self.episode_actions = [episode for episode in self.episode_actions if episode]
        # flatten the list of lists
        self.actions = [action for episode in self.episode_actions for action in episode]

    def analyze(self):
        """
        Analyzes the sequence of actions performed by the agent.
        :return:
        """
        
        # for each episode do a rolling window of X and output the maximal occuring action

        # for each episode do a rolling window of 5 and output the maximal occuring action
        rolling_window = 15
        rolled_episode_actions = []
        for episode in self.episode_actions:
            rolled_episodes = []
            for i in range(0, len(episode) - rolling_window):
                rolled = max(set(episode[i:i+rolling_window]), key=episode[i:i+rolling_window].count)
                rolled_episodes.append(rolled)
            rolled_episode_actions.append(rolled_episodes)
        #plot last 5 episodes
        print(rolled_episode_actions[-1])
        for episode in rolled_episode_actions[-5:]:
            plt.plot(episode)
        plt.show()

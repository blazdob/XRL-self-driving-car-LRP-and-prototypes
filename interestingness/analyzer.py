
from .analysis.state_frequency import StateFrequencyAnalysis
from .analysis.action_sequence import ActionSequenceAnalysis

class Analyzer():
    """
    Base class for all analyzers.
    """

    def __init__(self, states, actions):
        """
        :param str name: the name of the experiment.
        :param str output_dir: the path to the directory in which to save the results.
        """
        self.states = states
        self.actions = actions

    def analyze(self, analysis_type):
        """
        Analyzes the behavior of the agent and saves the results in the output directory.
        :param str analysis: the type of analysis to perform.
        :param str name: the name of the experiment.
        :param str output_dir: the path to the directory in which to save the results.
        :return:
        """
        if analysis_type == 'state_frequency':
            analyzer = StateFrequencyAnalysis(self.states)
            analyzer.analyze()
        elif analysis_type == 'action_sequence':
            analyzer = ActionSequenceAnalysis(self.actions)
            analyzer.analyze()
        else:
            raise ValueError('Unknown analysis: {}'.format(analysis_type))

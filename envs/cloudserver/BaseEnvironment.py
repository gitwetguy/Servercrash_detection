import os
from sklearn.preprocessing import MinMaxScaler

import Plots
import Utils

from gym import spaces
import gym


from Config import ConfigTimeSeries
import numpy as np



class TimeSeriesEnvironment:
    def __init__(self, config=ConfigTimeSeries(),
                 filename="test_df.csv", verbose=False,
                 scaler=MinMaxScaler()):
        """
        Initialization of one TimeSeries, should have suitable methods to prepare DataScience
        :param directory: Path to The Datasets
        :param config: Config class Object with specified Reward Settings
        :param filename: The Name of the specified TimeSeries we are working on
        :param verbose: if True, then prints __str__
        """
        self.filename = filename
        self.file = os.path.join(config.directory + self.filename)
        self.cfg = config
        self.sep = config.separator
        self.window = config.window
        self.scaler = scaler

        self.action_space_n = len(self.cfg.action_space)
        self.action_space = [0, 1]
        self.observation_space = spaces.Box(low=0., high=1.,
                                       shape=(1,config.window*2), dtype=np.float32)

        self.timeseries_cursor = -1
        self.timeseries_cursor_init = 0
        self.timeseries_states = []
        self.done = False
        self.timeseries_labeled = Utils.load_csv(self.file)

       

    def __info(self):
        """
        :return: String Representation of the TimeSeriesEnvironment Class, mainly for debug information
        """
        return "TimeSeries from: {}\n Header(labeled):\n {} \nHeader(unlabeled):\n {} \nRows:\n " \
               "{}\nMeanValue:\n {}\nMaxValue:\n {}\nMinValue:\n {}".format(
            self.filename,
            self.timeseries_labeled.head(
                3),
            self.timeseries_labeled.head(
                3),
            self.timeseries_labeled.shape[0],
            round(self.timeseries_labeled[self.cfg.value_columns].mean(), 2),
            round(self.timeseries_labeled[self.cfg.value_columns].max(), 2),
            round(self.timeseries_labeled[self.cfg.value_columns].min(), 2))

    def __state(self):
        """
        Every environment should implement a state function to return the current state
        :return: current state
        """
        pass

    def __reward(self, action):
        """
        Every environment should implement a reward function to return rewards for the current state and action chosen
        :return: reward
        """
        pass

    def reset(self):
        """
        Every environment should implement a reset method to return a initial State
        :return: init state
        """
        pass

    def step(self, action):
        """
        Every environment should implement a step method to return the S,A,R,S_,D tuples
        :return: state, action, reward, next_state, done
        """
        pass

    def update_cursor(self):
        """
        Increment the cursor
        :return: void
        """
        self.timeseries_cursor += 1

    def is_done(self):
        """
        Are we done with the current timeseries?
        :param cursor: position in dataframe
        :return: boolean
        """
        if self.timeseries_cursor >= len(self.timeseries_labeled) - 1:
            self.done = True
            return True
        else:
            self.done = False
            return False

    def normalize_timeseries(self):
        """
        Min,Max Normalization between 0,1
        :return: void
        """
        self.timeseries_labeled["value"] = self.scaler.fit_transform(self.timeseries_labeled[["value"]])

    def is_anomaly(self):
        """
        Is the current position a anomaly?
        :param cursor: position in dataframe
        :return: boolean
        """
        if self.timeseries_labeled['anomaly'][self.timeseries_cursor] == 1:
            return True
        else:
            return False

    def get_series(self, labelled=True):
        """
        Return the current series labelled or unlabelled
        :param labelled: boolean
        :return: pandas dataframe
        """
        
        return self.timeseries_labeled

    def __str__(self):
        """
        Get the current Filename if needed
        :return: String
        """
        return self.filename

    def __len__(self):
        """
        Get the length of the current dataframe
        :return: int
        """
        return self.timeseries_labeled.shape[0]


if __name__ == '__main__':
    ts = TimeSeriesEnvironment(verbose=True,
                               config=ConfigTimeSeries(),
                               filename="test_df.csv")
    ts.reset()
    count = 0
    
    while count < len(ts.timeseries_labeled) - 1:
        count += 1
    Plots.plot_series(ts.get_series(), name=str(ts))
    

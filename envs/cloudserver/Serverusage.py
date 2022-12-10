#!/usr/bin/env python3

import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv

import os
from sklearn.preprocessing import MinMaxScaler

import Plots
import Utils

from gym import spaces
import gym


from Config import ConfigTimeSeries
import numpy as np


class Serverusage(MetaEnv):
   
    def __init__(self, task=None):
        self.seed()
        
        self.config = ConfigTimeSeries()
        self.filename = self.config.filename
        self.file = os.path.join(self.config.directory + self.filename)
        self.sep = self.config.separator
        self.filename = self.config.filename
        self.window = self.config.window
        self.timeseries_labeled = Utils.load_csv(self.file)
        self.action_space_n = {0: 'normal', 1: 'abnormal'}
        self.state_dim = self.config.window*len(self.config.value_columns)
        low = np.zeros(self.state_dim,dtype=np.float32)
        high = np.ones(self.state_dim,dtype=np.float32)
        super(Serverusage, self).__init__(task)
        self.good_reward = 1
        self.bad_reward = 0.1
        
    
        self.observation_space = spaces.Box(low=low, high=high,shape=(self.config.window*len(self.config.value_columns),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_space_n))
        #print(self.action_space.sample())
        
        self.scaler = MinMaxScaler()
        
        self.timeseries_cursor = -1
        self.timeseries_cursor_init = 0
        self.timeseries_states = []
        self.done = False


        self.reset()

    # -------- MetaEnv Methods --------
    def sample_tasks(self, num_tasks):
        """
        Tasks correspond to a goal point chosen uniformly at random.
        """
        rd_arr = np.random.randint(self.config.window, size=num_tasks)
        goals = []

        for i in rd_arr:
            if np.sum(self.timeseries_labeled['anomaly'][i-self.window:i].values) >= 1:
                goals.append(1)
            else:
                goals.append(0)
 
        tasks = [{'goal': goal} for goal in goals]
       
        return tasks

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']

    # -------- Gym Methods --------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, env=True):
        """
        Reset the current Series to the first Value.
        :return: initial state
        """
        self.timeseries_cursor = self.timeseries_cursor_init
        #self.normalize_timeseries()
        self.done = False
        init_state = self.__state()
        return init_state
    
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

    def __state(self):
            """
            The Statefunction returning an array of the window states
            :return:
            """
            if self.is_done():
                return np.zeros(self.state_dim)

            # if self.timeseries_cursor >= self.window:
            win_state=[]
            for i in range(self.timeseries_cursor - self.window, self.timeseries_cursor):
                win_state.append(self.timeseries_labeled[self.config.value_columns].values[i + 1])
            win_state = np.array(win_state)
            win_state = self.scaler.fit_transform(win_state).reshape(-1,)
            # win_state = win_state.reshape(-1,)
            
            return win_state
            # else:
            #     win_state=[]
            #     for i in range(0, self.window):
            #         win_state.append(self.timeseries_labeled[self.config.value_columns].values[i + 1])
            #     win_state = np.array(win_state)
            #     win_state = self.scaler.fit_transform(win_state).reshape(-1,)
            #     # win_state = win_state.reshape(-1,)
                
            #     return win_state

    def step(self, action):
        """
        Taking a step inside the base environment with the action input
        :param action: certain action value
        :return: S,A,R,S_,D tuple
        """
        current_state = self.__state()
        reward = self.__reward(action)
        
        self.update_cursor()

        
        next_state = self.__state()
        

        #self._state, reward, done, self._task
        #return current_state, action, reward, next_state, self.is_done()

        return current_state, reward, self.is_done(), self._task

    

    def __reward(self, action):
        """
        The Rewardfunction returning rewards for certain actions in the environment
        :param action: type of action
        :return: arbitrary reward
        """

        # print(action)
        if self.timeseries_cursor >= self.window and not self.done:
            if np.sum(self.timeseries_labeled['anomaly'][self.timeseries_cursor-self.window:self.timeseries_cursor].values) >= 1:
                if action == 0:
                    return -self.bad_reward  # false negative, miss alarm
                else:
                    return self.good_reward  # 10      # true positive
            if np.sum(self.timeseries_labeled['anomaly'][self.timeseries_cursor-self.window:self.timeseries_cursor].values) == 0:
                if action == 1:
                    return -self.bad_reward
                else:
                    return self.good_reward
        return 0

    def render(self, mode=None):
        raise NotImplementedError
    
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
            round(self.timeseries_labeled[self.config.value_columns].mean(), 2),
            round(self.timeseries_labeled[self.config.value_columns].max(), 2),
            round(self.timeseries_labeled[self.config.value_columns].min(), 2))
    
    

if __name__ == '__main__':
    env = Serverusage()
    env.reset()
    #print(len(env.config.value_columns))
    idx = 1
    print(env.action_space.sample())
    print(env.action_space.sample())
    print(env.action_space.sample())
    print(env.action_space.sample())
    # while True:
    #     idx += 1
    #     s, r, d, t= env.step(1)
    #     print(s.shape)
    #     if d:
    #         print(idx)
    #         break
    

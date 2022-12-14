#!/usr/bin/env python3

import numpy as np
from gym import spaces
from gym.utils import seeding

from learn2learn.gym.envs.meta_env import MetaEnv

import os
from sklearn.preprocessing import MinMaxScaler

# import Plots
import Utils,glob

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
        self.filelist = glob.glob(os.path.join(self.config.directory+"pyod_res_ec2*"))
        self.sep = self.config.separator
        self.filename = self.config.filename
        self.window = self.config.window

        self.timeseries_set = []
        self.timeseries_labeled = Utils.load_csv(self.file)
        for pth in self.filelist:
            self.timeseries_set.append(Utils.load_csv(pth))

        self.action_space_n = {0: 'normal', 1: 'abnormal'}
        self.state_dim = len(self.config.value_columns)
        low = np.zeros(self.state_dim,dtype=np.float32)
        high = np.ones(self.state_dim,dtype=np.float32)
        super(Serverusage, self).__init__(task)
        self.good_reward = 1
        self.bad_reward = 1
        
    
        self.observation_space = spaces.Box(low=low, high=high,shape=(len(self.config.value_columns),), dtype=np.float32)
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
        # print(len(self.filelist))
        rd_dataidx = np.random.randint(len(self.filelist),size=1)
        self.timeseries_labeled=self.timeseries_set[rd_dataidx[0]]
        rd_arr = np.random.randint(self.timeseries_labeled.shape[0], size=num_tasks)
        
        # print(self.timeseries_set[rd_dataidx[0]])
        goals = []

        for i in rd_arr:
            if self.timeseries_labeled['Class'][i] == 1:
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
        if self.timeseries_cursor >= self.timeseries_labeled.shape[0] - 1:
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
            win_state=self.timeseries_labeled[self.config.value_columns].values[self.timeseries_cursor]
            
            # win_state = np.array(win_state)
            # win_state = self.scaler.fit_transform(win_state).reshape(-1,)
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

        if not self.done:
            if action == 1:
                if self.timeseries_labeled['Class'][self.timeseries_cursor]== 1:
                    return self.good_reward
                else:
                    return -self.bad_reward
            elif action == 0:
                if self.timeseries_labeled['Class'][self.timeseries_cursor]== 0:
                    return 0
                else:
                    return 0
        return 0

    def render(self, mode=None):
        raise NotImplementedError
    
    def is_anomaly(self):
        """
        Is the current position a anomaly?
        :param cursor: position in dataframe
        :return: boolean
        """
        if self.timeseries_labeled['Class'][self.timeseries_cursor] == 1:
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
    print(env.sample_tasks(num_tasks=10))
    print(env.get_series().describe())
    # while True:
    #     idx += 1
    #     s, r, d, t= env.step(1)
    #     print(s)
    #     if d:
    #         print(idx)
    #         break
    

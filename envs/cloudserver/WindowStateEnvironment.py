import sys
sys.path.append(r"D:\pythonwork\Servercrash_detection\envs\cloudserver")

from Config import ConfigTimeSeries
from BaseEnvironment import TimeSeriesEnvironment
import numpy as np
from learn2learn.gym.envs.meta_env import MetaEnv
from sklearn.preprocessing import MinMaxScaler
import numpy as np



cfg = ConfigTimeSeries()

class WindowStateEnvironment:
    """
    This Environment is sliding a window over the timeseries step by step. The window size can be configured
    The Shape of the states is therefore of the form (1, window_size).
    """

    def __init__(self, environment=TimeSeriesEnvironment(filename="test_df.csv", config=ConfigTimeSeries()), window_size=cfg.window,task=None):
        """
        Initialize the WindowStateEnvironment and wrapping the base environment
        :param environment: TimeSeriesEnvironment
        :param window_size: int
        """
        self.window_size = window_size
        self.columns = cfg.value_columns
        self.env = environment
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        self.env = environment
        if environment is None:
            raise TypeError("Base environment must be instantiated")
        elif isinstance(environment, TimeSeriesEnvironment):
            self.env = environment
        else:
            raise TypeError("Input is not of type TimeSeriesEnvironment")

    def __state(self):
        """
        The Statefunction returning an array of the window states
        :return:
        """
        if self.env.timeseries_cursor >= self.window_size:
            win_state=[]
            for i in range(self.timeseries_cursor - self.window_size, self.timeseries_cursor):
                win_state.append(self.env.timeseries_labeled[self.columns].values[i + 1])
            win_state = np.array(win_state)
            win_state = self.scaler.fit_transform(win_state).reshape(1,-1)
            
            return win_state
        else:
            return np.zeros(self.window_size)

    def __reward(self, action):
        """
        The Rewardfunction returning rewards for certain actions in the environment
        :param action: type of action
        :return: arbitrary reward
        """
        if self.timeseries_cursor >= self.window_size and not self.done:
            if np.sum(self.timeseries_labeled['anomaly'][self.timeseries_cursor-self.window_size:self.timeseries_cursor].values) >= 1:
                if action == 0:
                    return -5  # false negative, miss alarm
                else:
                    return 5  # 10      # true positive
            if np.sum(self.timeseries_labeled['anomaly'][self.timeseries_cursor-self.window_size:self.timeseries_cursor].values) == 0:
                if action == 1:
                    return -5
        return 0

    def reset(self):
        """
        Reset the current Series to the first Value.
        :return: initial state
        """
        self.env.timeseries_cursor = self.timeseries_cursor_init
        #self.normalize_timeseries()
        self.env.done = False
        init_state = self.__state()
        return init_state

    def step(self, action):
        """
        Taking a step inside the base environment with the action input
        :param action: certain action value
        :return: S,A,R,S_,D tuple
        """
        current_state = self.__state()
        reward = self.__reward(action)
        
        self.update_cursor()

        if self.is_done():
            next_state = []
        else:
            next_state = self.__state()

        #self._state, reward, done, self._task
        #return current_state, action, reward, next_state, self.is_done()
        return current_state, reward,  self.is_done(), self._task
    
    def sample_tasks(self, num_tasks):
        """
        Tasks correspond to a goal point chosen uniformly at random.
        """
        rd_arr = np.random.randint(self.window_size,size=num_tasks)
        goals = []

        for i in rd_arr:
            if np.sum(self.timeseries_labeled['anomaly'][i-self.window_size:i].values) >= 1:
                goals.append(1)
            else:
                goals.append(0)
 
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']

    def __len__(self):
        """
        Length of the current Timeseries
        :return: int
        """
        return len(self.env)

    def __getattr__(self, item):
        """
        Get the attribute of the base environment
        :param item: String of field key
        :return: attribute item of the base environment
        """
        return getattr(self.env, item)


if __name__ == '__main__':
    env = WindowStateEnvironment(
        TimeSeriesEnvironment(verbose=True, filename="test_df.csv", config=ConfigTimeSeries()))
    env.reset()
    idx = 1
    while True:
        idx += 1
        s, r, d, t= env.step(1)
        print(s, r, d, t)
        if d:
            print(idx)
            break

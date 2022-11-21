import sys
sys.path.append(r"D:\pythonwork\Servercrash_detection\envs\cloudserver")

from envs.cloudserver.BaseEnvironment import TimeSeriesEnvironment
from envs.cloudserver.WindowStateEnvironment import WindowStateEnvironment
from envs.cloudserver.Config import ConfigTimeSeries


env = WindowStateEnvironment(TimeSeriesEnvironment(verbose=True, filename="test_df.csv", config=ConfigTimeSeries()))
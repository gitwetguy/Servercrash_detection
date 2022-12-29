import pickle
import time
import pandas as pd
import sys
sys.path.append(r"D:\pythonwork\Servercrash_detection\envs\cloudserver")
from Config import ConfigTimeSeries
cfg = ConfigTimeSeries()

def pretty_print_df(dataframe, head=True):
    if head:
        return dataframe.head(5).to_markdown()
    return dataframe.to_markdown()


def store_object(object, filename):
    file = open(filename, "wb")
    pickle.dump(object, file)
    print("Successfully stored to {}".format(filename))


def load_object(filename):
    file = open(filename, "rb")
    object = pickle.load(file)
    print("Successfully loaded Object from {}".format(filename))
    return object


def start_timer():
    return time.time()


def get_duration(timer):
    return time.time() - timer


def load_csv(path, separator=",",columns=cfg.columns):
    
    return pd.read_csv(path, header=0, 
                           sep=separator,
                           encoding="utf-8")
   
     
    raise AttributeError("The path of the csv file might not be found.")

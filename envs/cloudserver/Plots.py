from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn import metrics



def plot_series(series, col=["CPU utilization (%)","Memory used  (%)"],name="time_series"):
    plt.figure(figsize=(15, 7))
    plt.plot(series[col])
    
    
    plt.title('Time Series: {}'.format(name))
    plt.xlabel('Timestamp')
    plt.ylabel('Usage')
    plt.show()


def plot_actions(actions, series):
    plt.figure(figsize=(15, 7))
    
    #print([idx for idx, element in enumerate(actions) if element==1])
    
    plt.scatter([idx for idx, element in enumerate(actions) if element==1], 
        series["value"].values[[idx for idx, element in enumerate(actions) if element==1]], 
        label="Actions", linewidths=6,zorder=3,marker='v',color="green")
    #plt.plot(series.index, series["anomaly"], label="True Label", linestyle="dotted")
    plt.scatter(series[series["anomaly"]==1].index, 
        series[series["anomaly"]==1]["value"], 
        label="True Label",linewidths=10,c='red',zorder=2)
    plt.plot(series.index, series["value"], label="Series", linestyle="dashed",zorder=1,color="blue")
    plt.legend()
    plt.ylabel('Reward Sum')
    plt.savefig("./actions_res.jpg")
    plt.show()
    


def plot_learn(data):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=data,
    ).set_title("Learning")
    plt.ylabel('Reward Sum')
    plt.show()


def plot_reward(result):
    plt.figure(figsize=(15, 7))
    sb.lineplot(
        data=result,
    ).set_title("Reward Random vs Series")
    plt.ylabel('Reward Sum')
    plt.show()


def evaluation_func(actions, series):
    
    true = series["anomaly"].values
    prediction = actions
    print('Precision:', round(metrics.precision_score(true, prediction),3))
    print('Recall:', round(metrics.recall_score(true, prediction),3))
    print('F1:', round(metrics.f1_score(true, prediction),3))


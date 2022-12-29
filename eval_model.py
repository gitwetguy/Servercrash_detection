import sys,os,gym
import numpy as np
sys.path.append(r"D:\pythonwork\Servercrash_detection\envs\cloudserver")
from envs.cloudserver.Utils import load_csv

from policy.policies import CategoricalPolicy
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix



from scipy import stats
ts_data = load_csv(r"D:\pythonwork\Servercrash_detection\dataset\test_df.csv")


def gen_dataset(data,x_window_size,y_window_size):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        scaler = MinMaxScaler()
        X_train.append(scaler.fit_transform(data.iloc[i-x_window_size:i, 1].values.reshape(-1,1)))
        
        
        if np.sum(data.iloc[i-x_window_size:i, -1].values) >= 1:
            y_train.append(1)
        else:
            y_train.append(0)

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    
    #X_train = np.reshape(X_train, (X_train.shape[0], x_window_size,data.shape[1] ))

    """if y_window_size == 1:
        y_train = np.reshape(y_train,(y_train.shape[0], y_train.shape[2]))
    else:
        y_train = np.reshape(y_train,(y_train.shape[0],y_window_size,y_train.shape[2]))"""
        
    print("Gen data info:")
    print("X_data_shape:{}".format(X_train.shape))
    print("y_data_shape:{}".format(y_train.shape))
    print("\n")
          
    return X_train,y_train


    
 


#load model
save_path = "save"
exp_num = 27
device = "cuda"
load_model_path = os.path.join(save_path,"exp{}".format(exp_num),"it{}_model.pt".format(exp_num))
policy = CategoricalPolicy(10,2,device='cuda')
policy.load_state_dict(torch.load(load_model_path))
policy.eval().to(device)


def predict(model,test_data,y_test_data):
    
    # model is self(VGG class's object)
    
    count = test_data.shape[0]
    result_np = []
        
    for idx in range(0, count):
        # print(idx)
        
        input_data = torch.Tensor(test_data[idx].reshape(-1,)).to(device)

        # print(img.shape)
        ac,_ = model(input_data)
        
        pred_np = ac.cpu().numpy()
        # for elem in pred_np:
        result_np.append(pred_np)
        # result_np = np.array(result_np)
    return result_np

X_test,y_test = gen_dataset(ts_data,10,1)
y_pred = predict(policy,X_test,y_test)
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score,accuracy_score,precision_score,precision_recall_curve,recall_score

print("accuracy_score: {}".format(accuracy_score(y_test, y_pred)))
print("precision_score: {}".format(precision_score(y_test, y_pred)))
print("recall_score: {}".format(recall_score(y_test, y_pred)))
print("f1_score: {}".format(f1_score(y_test, y_pred)))


import matplotlib.pyplot as plt
import seaborn as sn
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
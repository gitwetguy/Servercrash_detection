import pandas as pd 
import numpy as np
# import tsfel
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.loda import LODA

import matplotlib.pyplot as plt
from numpy import percentile
from sklearn.preprocessing import MinMaxScaler
import glob,os


path_list = glob.glob(r"D:\pythonwork\Servercrash_detection\dataset\ec2*.csv")
# print(os.path.basename(path_list[0]).split(".")[0])
# os.path.basename(path_list[i]).split(".")[0]



df_list = []
for path in path_list:
    temp_df = pd.read_csv(path).iloc[:,2:]
    df_list.append(temp_df)
    
    



x_window_size = 20
y_window_size = 1
random_state = 42 

for idx,df in enumerate(df_list):
    print(path_list[idx])
    X_train = []   #預測點的前 60 天的資料
    y_train = [] 
    for i in range(x_window_size, df.shape[0]-y_window_size,y_window_size):
            X_train.append(df.iloc[i-x_window_size:i, 0].values)
            
            if np.sum(df.iloc[i-x_window_size:i, -1].values) >= 1:
                y_train.append(1)
            else:
                y_train.append(0)

    X_train, y_train = np.array(X_train), np.array(y_train)


    #CLF
    ot = np.asarray(np.where(y_train == 1)).shape[0]
    outliers_fraction = ot/ y_train.shape[0]

    classifiers = {
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        "Lightweight on-line detector of anomalies":LODA(contamination=outliers_fraction)}

    col = []
    for i, clf in enumerate(classifiers.keys()):
        print('Model', i + 1, clf)
        col.append(clf)


    res_df = pd.DataFrame()
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print("Fitting {} {}".format(i,clf_name))

        feature_data = []
        for i in range(0,X_train.shape[0]):
            X = X_train[i]
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.reshape(-1,1))
            clf.fit(X)
            y_train_scores = clf.decision_scores_
            # X = np.append(X,y_train[i])
            feature_data.append(np.mean((y_train_scores)))
            # print(np.sum((y_train_scores)),y_train[i])
        res_df[clf_name]=feature_data

    res_df["Class"]=y_train
    
    res_df.to_csv("dataset/pyod_res_{}.csv".format(os.path.basename(path_list[idx]).split(".")[0]))
    










# cfg = tsfel.get_features_by_domain("statistical")
# X = tsfel.time_series_features_extractor(cfg, X_train[1].reshape(-1,))


# col=X.columns.values
# col = np.append(col,"Class")
# print(col)
# feature_data = []

# for i in range(0,X_train.shape[0]):
#     X = tsfel.time_series_features_extractor(cfg, X_train[i].reshape(-1,)).values
#     X = np.append(X,y_train[i])
#     feature_data.append(X)
# res_df = pd.DataFrame(np.array(feature_data).reshape(-1,len(col)),
#                    columns=col)

# res_df.to_csv("statistical_res.csv")
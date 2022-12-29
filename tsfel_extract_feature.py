import pandas as pd 
import numpy as np
import tsfel
from sklearn.preprocessing import MinMaxScaler

# import pyodpip install pyod
df = pd.read_csv(r"D:\pythonwork\Servercrash_detection\dataset\test_df.csv")
print(df)
x_window_size = 20
y_window_size = 1

X_train = []   #預測點的前 60 天的資料
y_train = [] 
for i in range(x_window_size, df.shape[0]-y_window_size,y_window_size):
        X_train.append(df.iloc[i-x_window_size:i, 1:3].values)
        
        if np.sum(df.iloc[i-x_window_size:i, -1].values) >= 1:
            y_train.append(1)
        else:
            y_train.append(0)

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(X_train)

cfg = tsfel.get_features_by_domain("statistical")
X = tsfel.time_series_features_extractor(cfg, X_train[1].reshape(-1,))


col=X.columns.values
col = np.append(col,"Class")
print(col)
feature_data = []

for i in range(0,X_train.shape[0]):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_train[i])
    X = tsfel.time_series_features_extractor(cfg, X.reshape(-1,)).values
    print(X_train[i].reshape(-1,).shape,X.shape)
    X = np.append(X,y_train[i])
    feature_data.append(X)
res_df = pd.DataFrame(np.array(feature_data).reshape(-1,len(col)),
                   columns=col)

res_df.to_csv("statistical_res.csv")
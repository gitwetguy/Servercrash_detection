import os

class ConfigTimeSeries:
    def __init__(self, normal=0, 
                       anomaly=1, 
                       action_space=[0, 1], 
                       separator=",", 
                       home="D:/pythonwork/Servercrash_detection", 
                       window=10):
        #D:\pythonwork\Servercrash_detection\dataset\test_df.csv
        self.action_space = action_space
        self.directory = os.path.join(home+"/dataset")+"/"
        self.anomaly = anomaly
        self.normal = normal
        self.separator = separator
        self.window = window
        self.columns = ["One-class SVM (OCSVM)"	,"Isolation Forest"	,"Lightweight on-line detector of anomalies"	,"Class"]
        self.value_columns = ["One-class SVM (OCSVM)"	,"Isolation Forest"	,"Lightweight on-line detector of anomalies"]
        self.filename = "test_df.csv"
        

    def __repr__(self):
        return {"normal": self.normal, "anomaly": self.anomaly,
                "action_space": self.action_space}

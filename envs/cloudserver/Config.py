class ConfigTimeSeries:
    def __init__(self, normal=0, anomaly=1, action_space=[0, 1], separator=",", directory="D:/pythonwork/Servercrash_detection/dataset/", window=25):
        #D:\pythonwork\Servercrash_detection\dataset\test_df.csv
        self.action_space = action_space
        self.directory = directory
        self.anomaly = anomaly
        self.normal = normal
        self.separator = separator
        self.window = window
        self.columns = ["dt","CPU utilization (%)","Memory used  (%)","anomaly"]
        self.value_columns = ["CPU utilization (%)","Memory used  (%)"]

    def __repr__(self):
        return {"normal": self.normal, "anomaly": self.anomaly,
                "action_space": self.action_space}
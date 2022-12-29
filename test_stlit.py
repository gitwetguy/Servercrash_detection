import streamlit as st
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import asyncio
import aiohttp
import numpy as np

st.set_page_config(layout="wide")


# lof_url = 'https://ts-anomaly-detection-lof.herokuapp.com/'
# if_url = 'https://ts-anomaly-detection-if.herokuapp.com/'
# stl_url = 'https://ts-anomaly-detection-stl.herokuapp.com/'


# org_df = pd.read_csv(r"D:\pythonwork\Servercrash_detection\dataset\test_df.csv")
# print(org_df.describe())
# x_window_size = 20
# y_window_size = 1
# random_state = 42 

# X_train = []   #預測點的前 60 天的資料
# y_train = [] 
# for i in range(x_window_size, org_df.shape[0]-y_window_size,y_window_size):
#         X_train.append(org_df.iloc[i-x_window_size:i, 1:3].values)
        
#         if np.sum(org_df.iloc[i-x_window_size:i, -1].values) >= 1:
#             y_train.append(1)
#         else:
#             y_train.append(0)

# X_train, y_train = np.array(X_train), np.array(y_train)


def save_uploadedfile(uploadedfile):

    with open(os.path.join("data/uploads",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())

def visualize_sensor_data(org_df):
    dfSensors = org_df.drop(['Class'], axis=1)
    sensorNames=dfSensors.columns
    for sensor in sensorNames:
        sns.set_context('talk')
        fig = plt.figure(figsize=(24,8))
        #_ = plt.plot(dfBroken[sensor], linestyle='none', marker='X', color='red', markersize=12)
        _ = plt.plot(org_df[sensor], color='black')
        _ = plt.title(sensor)
        st.pyplot(fig)

def visualize_actual_anomalies(org_df, dfBroken):
    dfSensors = org_df.drop(['Class'], axis=1)
    sensorNames=dfSensors.columns
    for sensor in sensorNames:
        sns.set_context('talk')
        fig = plt.figure(figsize=(24,8))
        _ = plt.plot(dfBroken[sensor], linestyle='none', marker='X', color='red', markersize=12)
        _ = plt.plot(org_df[sensor], color='blue')
        _ = plt.title("Model Query Anomaly : "+sensor)
        st.pyplot(fig)

def visualize_common_anomalies(all_anomalies_df, dfBroken):
    #isolation forest
    anomalies_df_if  = all_anomalies_df[0]
    a = anomalies_df_if.loc[anomalies_df_if['PredictedAnamoly'] == -1] #anomaly
    a = a.loc[a['machine_status']=='BROKEN']

    #LOF
    anomalies_df_lof  = all_anomalies_df[1]
    b = anomalies_df_lof.loc[anomalies_df_lof['PredictedAnamoly'] == -1] #anomaly
    b = b.loc[b['machine_status']=='BROKEN']

    #plot the graph
    fig = plt.figure(figsize=(16,5))
    _ = plt.plot(anomalies_df_if['sensor_values'], color='grey', label='Normal')
    #_ = plt.plot(dfBroken['sensor_values'], linestyle='none', marker='X', color='red', label='Broken', markersize=25)

    _ = plt.plot(a['sensor_values'], linestyle='none', marker='o', color='black', label='Forest', markersize=15)
    _ = plt.plot(b['sensor_values'], linestyle='none', marker='*', color='white', label='LOF', markersize=15)
    _ = plt.xlabel('Date and Time')
    _ = plt.ylabel('Sensor Reading')
    _ = plt.title('sensor_00 iForestVsLOF Anomalies')
    _ = plt.legend(loc='best')
    st.pyplot(fig)


# async def make_async_requests(urls, file_details):
#     tasks = []

#     for url in urls:
#         #make fastapi request to get anomalies
#         tasks.append(make_async_api_call(url, file_details))
    
#     results = await asyncio.gather(*tasks)
#     #return list(chain(*(res for res in results)))
#     return results

# async def make_async_api_call(url, file_details):
#     file_url = './data/uploads/'+file_details['FileName']
#     files = {'sensor_data': open(file_url, 'rb').read()}
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, data = files) as res:
#             data = await res.json()
#             return data

def make_api_call(url, file_details):
    files = {'sensor_data': open('./data/uploads/'+file_details['FileName'], 'rb')}
    res = requests.post(url, files=files)
    return res.json()

def plot_anomalies(anomalies_df, dfBroken, model):
    
    #plot detected anomalies vs the actual anomalies in the same graph
    st.subheader("Detected Anomalies by "+model)
    a = anomalies_df.loc[anomalies_df['PredictedAnamoly'] == -1] #anomaly
    a = a.loc[a['machine_status']=='BROKEN']

    fig = plt.figure(figsize=(24,8))
    _ = plt.plot(anomalies_df['sensor_values'], color='grey', label='Normal')
    _ = plt.plot(a['sensor_values'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly Detected', alpha=0.3)
    #_ = plt.plot(dfBroken['sensor_values'], linestyle='none', marker='X', color='red', markersize=12, label='Broken')
    _ = plt.xlabel('Date and Time')
    _ = plt.ylabel('Sensor Reading')
    _ = plt.title('sensor FOREST ISOLATION Anomalies')
    _ = plt.legend(loc='best')
    st.pyplot(fig)

    return anomalies_df

def plot_anamoly_graph(dfsensorIndexed, dfanomali):
    
    st.subheader("Detected Anomalies by STL Decomposition")

    fig = plt.figure(figsize=(30, 8))
    _ = plt.plot_date(dfsensorIndexed.index, dfsensorIndexed.iloc[:, 0], linestyle='--', zorder=1)
    _ = plt.scatter(dfanomali.index, dfanomali.iloc[:, 0], color='r', marker='X', zorder=2, s=250)
    _ = plt.xticks(rotation=270)
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Sampled Sensor Reading')
    _ = plt.title('STL decomposition  Anomalies')
    _ = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    st.pyplot(fig)


def main():
    st.title("Anomaly Detection in Cloud Web App")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    

    if st.button("Check Anomalies"):
        if uploaded_file is not None:
            
            # #uploaded file details
            # file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}

            # #save the uploaded file
            # save_uploadedfile(uploaded_file)
            # # org_df  = pd.read_csv(uploaded_file)
            org_df = pd.read_csv(r"D:\pythonwork\Servercrash_detection\dataset\test_df.csv")
            #set timestamp as index
            org_df['dt'] = pd.to_datetime(org_df['dt'])
            org_df = org_df.set_index('dt')
            
            #Actual Sensor Data
            st.subheader("Actual Usage Data")
            visualize_sensor_data(org_df)
            
            #Actual Anomalies
            st.subheader("Model Query")
            dfBroken = org_df[org_df['Class']==1]
            visualize_actual_anomalies(org_df, dfBroken)

            
            
            # start = time.time()

            # all_anomalies_df = []
            # #make api calls and plot graphs
            # results = asyncio.run(make_async_requests(urls, file_details))
            # for i in range(len(results)):
            #     if models[i]=="STL":
            #         dict = results[i]

            #         df_anomalies_list = dict['anomalies_df_list']
            #         df_anomalies = pd.DataFrame(df_anomalies_list, columns =['timestamp', 'sensor_values', 'resid'])
            #         #set timestamp as index
            #         df_anomalies['timestamp'] = pd.to_datetime(df_anomalies['timestamp'])
            #         df_anomalies = df_anomalies.set_index('timestamp')

            #         df_sensor_list = dict['sensor_df_list']
            #         df_sensor = pd.DataFrame(df_sensor_list, columns =['timestamp', 'sensor_values'])
            #         #set timestamp as index
            #         df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
            #         df_sensor = df_sensor.set_index('timestamp')
                    
            #         plot_anamoly_graph(df_sensor, df_anomalies)
            #     else:
            #         dict = results[i]
            #         df_list = dict['anomalies_df']
   
            #         df_anomalies = pd.DataFrame(df_list, columns =['timestamp', 'sensor_values', 'machine_status', 'PredictedAnamoly'])
            #         #set timestamp as index
            #         df_anomalies['timestamp'] = pd.to_datetime(df_anomalies['timestamp'])
            #         df_anomalies = df_anomalies.set_index('timestamp')

            #         anomalies_df = plot_anomalies(df_anomalies, dfBroken, models[i])
            #         all_anomalies_df.append(anomalies_df)
            
            # #common anomalies plot
            # if len(all_anomalies_df) > 1:
            #     st.subheader("Common Anomalies")
            #     visualize_common_anomalies(all_anomalies_df, dfBroken)
            
            # end = time.time()
            # total_time = end - start
            # st.write("Total Time taken to execute API Call(s): "+str(total_time))
            st.subheader("Analyst verify")
            st.button("verify anomaly")

                

if __name__ == '__main__':
    main()
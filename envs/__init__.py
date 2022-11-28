#!/usr/bin/env python3

from gym.envs.registration import register
import sys
import gym
sys.path.append(r"D:\pythonwork\Servercrash_detection")



# 2D Navigation
# ----------------------------------------
#D:\pythonwork\Servercrash_detection\envs\cloudserver\Serverusage.py
register(
    'Serverusage',
    entry_point='envs.cloudserver.Serverusage:Serverusage',
    max_episode_steps=100
)
#D:\pythonwork\learn2learn\learn2learn\gym\envs\cloudserver\cloudserver_ad.py
# Server Crash prediction
# ----------------------------------------

# env = gym.make("Serverusage")
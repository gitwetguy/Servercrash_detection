#!/usr/bin/env python3

"""
Trains a 2-layer MLP with MAML-TRPO.

Usage:

python examples/rl/maml_trpo.py
"""

import random
from copy import deepcopy
import sys,os
sys.path.append(r"D:\pythonwork\Servercrash_detection\envs\cloudserver")

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import learn2learn as l2l
from policy.policies import CategoricalPolicy

from envs.cloudserver.Serverusage import Serverusage
from envs.cloudserver.BaseEnvironment import TimeSeriesEnvironment
from envs.cloudserver.WindowStateEnvironment import WindowStateEnvironment
from envs.cloudserver.Config import ConfigTimeSeries

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from envs.cloudserver.Utils import load_csv
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime






def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    
    log_probs = learner.log_prob(states, actions)
    # print(log_probs)
    # print("states:{}".format(states.shape))
    # print("states:{}".format(states))
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    #cal loss on train task
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    #cal grad
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='Surrogate Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl




def gen_dataset(data):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    X_train = data.iloc[:,:-1].values
    y_train = data.iloc[:,-1].values.reshape(-1,1)
    
        
    print("Gen data info:")
    print("X_data_shape:{}".format(X_train.shape))
    print("y_data_shape:{}".format(y_train.shape))
    print("\n")
          
    return X_train,y_train

def predict(model,test_data,y_test_data):
    
    # model is self(VGG class's object)
    
    count = test_data.shape[0]
    result_np = []
        
    for idx in range(0, count):
        # print(idx)
        
        input_data = torch.Tensor(test_data[idx].reshape(-1,)).to("cuda")

        # print(img.shape)
        ac,_ = model(input_data)
        
        pred_np = ac.cpu().numpy()
        # for elem in pred_np:
        result_np.append(pred_np)
        # result_np = np.array(result_np)
    return result_np

def evaluate_model(model,X_test,y_test):
    
    model.eval().to("cuda")
    res = predict(model,X_test,y_test)
    from sklearn.metrics import f1_score,accuracy_score,precision_score,precision_recall_curve,recall_score

    print("accuracy_score: {}".format(accuracy_score(y_test, res)))
    print("precision_score: {}".format(precision_score(y_test, res)))
    print("recall_score: {}".format(recall_score(y_test, res)))
    print("f1_score: {}".format(f1_score(y_test, res)))

    return accuracy_score(y_test, res),precision_score(y_test, res),recall_score(y_test, res),f1_score(y_test, res)


def main(
        env_name='Serverusage',
        exp_name = "pyod_feature_2adapt_steps_meta_bsz20_adapt_bsz20_TNFN_reward0",
        hidden_layer=[100,100],
        #lr in inner loop
        adapt_lr=0.001,
        #lr in outer loop
        meta_lr=1.0,
        #adapt in outer loop
        adapt_steps=2,
        num_iterations=2000,
        #how many task in inner loop
        meta_bsz=10,
        #how many gd dec in inner loop
        adapt_bsz=10,
        tau=1.00,
        gamma=0.95,
        seed=42,
        num_workers=10,
        cuda=1,
):

    ts_data = load_csv(r"D:\pythonwork\Servercrash_detection\dataset\pyod_res.csv")
    X_test,y_test = gen_dataset(ts_data)
    

    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    writer = SummaryWriter()
    

    device_name = 'cpu'
    if cuda:
        torch.cuda.manual_seed(seed)
        device_name = 'cuda'
    device = torch.device(device_name)

    def make_env():
        
        env = gym.make(env_name)
        # env = ch.envs.ActionSpaceScaler(env)
        return env
    
    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    cfg = ConfigTimeSeries()
    
    print("main size :{}--{}".format(env.state_size, env.action_size))
    policy = CategoricalPolicy(env.state_size,env.action_size,hiddens=hidden_layer,device=device)
    if cuda:
        policy = policy.to(device)
        print(summary(policy, input_size=(len(cfg.value_columns),)))

    
    #print(env.state_size, env.action_size)
    baseline = LinearValue(env.state_size, env.action_size)
    result_seq = []

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H%M%S")

    exp_num = len(os.listdir("runs"))
    print("Exp {} \n".format(exp_num))
    savemodel_path = "save/exp{}".format(exp_num)
    isExist = os.path.exists(savemodel_path)
    if not isExist:
   # Create a new directory because it does not exist
        os.makedirs(savemodel_path)
        print("[{}] is created!".format(savemodel_path))

    for iteration in range(num_iterations):
        
        
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(clone, episodes=adapt_bsz)
                if cuda:
                    train_episodes = train_episodes.to(device, non_blocking=True)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)
        result_seq.append(adaptation_reward)
        
        writer.add_scalar("Reward/{}_exp{}-{}".format(exp_name,exp_num,dt_string), adaptation_reward, iteration)
        writer.close()
        
        # TRPO meta-optimization
        backtrack_factor = 0.5
        ls_max_steps = 15
        max_kl = 0.01
        if cuda:
            policy = policy.to(device, non_blocking=True)
            baseline = baseline.to(device, non_blocking=True)
            iteration_replays = [[r.to(device, non_blocking=True) for r in task_replays] for task_replays in
                                 iteration_replays]

        # Compute CG step direction
        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma,
                                               adapt_lr)
        grad = autograd.grad(old_loss,
                             policy.parameters(),
                             retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line-search
        for ls_step in range(ls_max_steps):
            
            stepsize = backtrack_factor ** ls_step * meta_lr
            clone = deepcopy(policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(-stepsize, u.data)
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, tau, gamma,
                                               adapt_lr)
            # print(clone)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                break

        acc,pre,rc,f1 = evaluate_model(clone,X_test,y_test)
        writer.add_scalar("Accuracy/{}_exp{}-{}".format(exp_name,exp_num,dt_string), acc, iteration)
        writer.add_scalar("Precision/{}_exp{}-{}".format(exp_name,exp_num,dt_string), pre, iteration)
        writer.add_scalar("Recall/{}_exp{}-{}".format(exp_name,exp_num,dt_string), rc, iteration)
        writer.add_scalar("F1 score/{}_exp{}-{}".format(exp_name,exp_num,dt_string), f1, iteration)
        writer.close()

        torch.save(policy.state_dict(), "{}/it{}_model.pt".format(savemodel_path,exp_num))
        np.savetxt("{}/it{}_reward.csv".format(savemodel_path,exp_num), np.array(result_seq), delimiter=",")

    
    
               

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import math

import cherry as ch
import torch
from torch import nn
from torch.distributions import Normal, Categorical

EPSILON = 1e-6


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class CaviaDiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', num_context_params=2, device='cpu'):
        super(CaviaDiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size+num_context_params, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mean = nn.Sequential(*layers).to(self.device)
        self.sigma = nn.Parameter(torch.Tensor(output_size)).to(self.device)
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        state = state.to(self.device, non_blocking=True)
        # concatenate context parameters to input
        state = torch.cat((state, self.context_params.expand(state.shape[:-1] + self.context_params.shape)),
                          dim=len(state.shape) - 1)

        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):

        density = self.density(state)
        action = density.sample()
        return action

    def reset_context(self):
        self.context_params[:] = 0  # torch.zeros(self.num_context_params, requires_grad=True).to(self.device)


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', device='cpu'):
        super(DiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        state = state.to(self.device, non_blocking=True)
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class CategoricalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, device='cpu'):
            super(CategoricalPolicy, self).__init__()
            self.device = device
            if hiddens is None:
                hiddens = [100, 100]
            layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
            for i, o in zip(hiddens[:-1], hiddens[1:]):
                layers.append(linear_init(nn.Linear(i, o)))
                layers.append(nn.ReLU())
                
            layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
            layers.append(nn.Softmax())
            self.mean = nn.Sequential(*layers)

            self.input_size = input_size
            self.sigma = nn.Parameter(torch.Tensor(output_size))
            self.sigma.data.fill_(math.log(1))

    def forward(self, state):
        state=state.to(self.device, non_blocking=True)
        #state = ch.onehot(state, dim=self.input_size)
        
        loc = self.mean(state).to(self.device)
        # print(loc)
        density = Categorical(logits=loc)
        # print(density)
        action = density.sample()
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return action, {'density': density, 'log_prob': log_prob}
    
    def density(self, state):
        # state = state.to(self.device, non_blocking=True)
        state = state.to(self.device)
        # print("state:{}".format(state.shape))
        loc = self.mean(state)
        # print("loc:{}".format(loc.shape))
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        try:
           Normal(loc=loc, scale=scale)
        except:
            torch.save(loc, r'D:\pythonwork\Servercrash_detection\error\loc.pt')
            torch.save(scale, r'D:\pythonwork\Servercrash_detection\error\scale.pt')
            torch.save(state, r'D:\pythonwork\Servercrash_detection\error\state.pt')
            print(loc,scale) 
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    # def forward(self, state):
    #     density = self.density(state)
    #     action = density.sample()
    #     return action
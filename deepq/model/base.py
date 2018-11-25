from abc import abstractmethod, ABCMeta
from six import with_metaclass
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
import torch.optim as optim
from ..base import torchify, torchify32, numpify, constant
from toolz import last
import numpy as np

class Model(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def evaluate(self, state):
        '''
        Return a vector of action values for the given state(s).
        '''

    @abstractmethod
    def learn(self, state, action, reward, next_state, done, weight):
        '''
        Update parameters based on given data.
        
        Parameters
        ==========
        
        state (ndarray, dtype=float, shape=(n,state_size)): The states for the learning 
            sample.
        
        action (ndarray, dtype=int, shape=n): The chosen actions for the learning sample.
        
        reward (ndarray, dtype=float, shape=n): The received rewards for the learning 
            sample.
        
        done (ndarray, dtype=bool, shape=n): Boolean vector indicating whether each 
            learning experience led to a terminal state.
        
        next_state (ndarray, dtype=float, shape=(n,state_size)): The states for the 
            learning sample after the chosen actions were taken.
        
        
        Returns
        =======
        
        error (ndarray, dtype=float, shape=n): The error for each learning experience.
        '''
    
    @abstractmethod
    def register_progress(self, agent):
        '''
        Inform the model about progress by the agent.  Should be called by the agent
        after each call to learn.  Allows the model to adjust learning parameters.
        '''

class DoubleNetworkModel(Model):
    def __init__(self, network, optimizerer=partial(optim.Adam, lr=5e-4), 
                 schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)), 
                 gamma=.9, tau=1., window=100):
        self.q_local = network
        self.q_target = deepcopy(self.q_local)
        self.optimizer = optimizerer(self.q_local.parameters())
        self.scheduler = schedulerer(self.optimizer)
        self.gamma = gamma
        self.tau = tau
        self.window = window
    
    def evaluate(self, state):
        return self.q_local.forward(torchify32(state))
        
    def soft_update(self, q_local, q_target, tau):
        for target_param, local_param in zip(q_target.parameters(), q_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    @abstractmethod
    def target(self, reward, next_state, done):
        pass
    
    def learn(self, state, action, reward, next_state, done, weight):
        target = self.target(reward, next_state, done)
        prediction = self.q_local.forward(torchify32(state)).gather(1, torchify(action).unsqueeze(1))
        
        weight_ = torchify32(weight).unsqueeze(1)
        loss = F.mse_loss(weight_ * prediction, weight_ * target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.q_local, self.q_target, self.tau)
        
        return numpify((target - prediction).squeeze())
    
    def register_progress(self, agent):
        if len(agent.train_scores) > self.window:
            average_reward = np.mean(list(map(last, agent.train_scores[-self.window:])))
            self.scheduler.step(average_reward)
        
        
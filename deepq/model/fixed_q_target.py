from .base import Model
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
import torch.optim as optim
import torch
from ..base import torchify, torchify32, numpify
import numpy as np

class FixedQTargetModel(Model):
    def __init__(self, network, optimizerer=partial(optim.Adam, lr=5e-4), gamma=.9, tau=1.):
#         self.device = (
#                        device if device is not None else 
#                        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#                        )
        self.q_local = network#.to(self.device)
        self.q_target = deepcopy(self.q_local)#.to(self.device)
        self.optimizer = optimizerer(self.q_local.parameters())
        self.gamma = gamma
        self.tau = tau
    
    def evaluate(self, state):
        return self.q_local.forward(torchify32(state))
        
    def soft_update(self, q_local, q_target, tau):
        for target_param, local_param in zip(q_target.parameters(), q_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def learn(self, state, action, reward, next_state, done, weight):
        q_target_output = (torchify32(~done).unsqueeze(1) * 
            self.q_target.forward(torchify32(next_state)).detach().max(1)[0].unsqueeze(1))
        target = torchify32(reward).unsqueeze(1) + self.gamma * q_target_output
        prediction = self.q_local.forward(torchify32(state)).gather(1, torchify(action).unsqueeze(1))
        
        weight_ = torchify32(weight).unsqueeze(1)
        loss = F.mse_loss(weight_ * prediction, weight_ * target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.q_local, self.q_target, self.tau)


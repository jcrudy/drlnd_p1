from .base import Policy
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon_start, epsilon_decay, epsilon_min):
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reset()
    
    def reset(self):
        self.epsilon = self.epsilon_start
    
    def choose(self, values):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(values))
        else:
            return np.random.choice(np.max(values) == values)
        
        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

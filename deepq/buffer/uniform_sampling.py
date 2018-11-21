from .base import ReplayBuffer
import numpy as np

class UniformSamplingReplayBuffer(ReplayBuffer):
    def sample_indices(self, sample_size):
        return np.random.choice(range(len(self)), size=sample_size), np.ones(shape=sample_size) / float(sample_size)
    
    def report_errors(self, indices, errors):
        pass
    
    def append(self, experience):
        self.buffer.append(experience)

    def __getitem__(self, indices):
        return tuple(self.buffer[idx] for idx in indices)
    
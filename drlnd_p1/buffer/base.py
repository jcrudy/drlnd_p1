from collections import deque
from abc import abstractmethod

class ReplayBuffer(object):
    def __init__(self, action_size, buffer_size):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
#     @abstractproperty
#     def buffer(self):
#         '''
#         A container for Experiences.  Must implement __len__, be indexable by 
#         non-negative integers, and be iterable.
#         '''
    
    def __len__(self):
        return len(self.buffer)
    
    @abstractmethod
    def append(self, experience):
        '''
        Append an Experience to the ReplayBuffer for later sampling.
        '''
    
    @abstractmethod
    def sample_indices(self, sample_size):
        '''
        Return a tuple of the indices of a sample from the ReplayBuffer and 
        the associated probabilities.
        '''
    
    @abstractmethod
    def __getitem__(self, indices):
        '''
        Return tuple of Experiences from the buffer at the given indices.
        '''
    
    @abstractmethod
    def report_errors(self, indices, errors):
        '''
        Inform the ReplayBuffer of errors from training on the given indices.
        '''

from abc import abstractmethod, ABCMeta
from six import with_metaclass

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
        '''
from .base import Environment, ClosedEnvironmentError
from unityagents import UnityEnvironment
from abc import abstractclassmethod
from . import resources

class UnityBasedEnvironment(Environment):
    @abstractclassmethod
    def path(self):
        '''
        Subclasses should set this to be the path to the 
        desired Unity environment.
        '''

    def __init__(self, graphics=False):
        self.env = UnityEnvironment(file_name=self.path, no_graphics=(not graphics))
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        example_state = env_info.vector_observations[0]
        self._state_size = len(example_state)
        self._n_actions = self.brain.vector_action_space_size
        self.closed = False
    
    def reset(self, train):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        return env_info.vector_observations[0]
    
    def step(self, action):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.step(action)[self.brain_name]
        state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0] 
        return state, reward, done
    
    def close(self):
        if self.closed:
            return
        self.env.close()
        self.closed = True
    
class BananaEnvironment(UnityBasedEnvironment):
    path = resources.banana
    
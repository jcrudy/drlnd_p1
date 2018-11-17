from abc import abstractmethod


class Model(object):
    @abstractmethod
    def evaluate(self, state):
        '''
        Return a vector of action values for the given state(s).
        '''

    @abstractmethod
    def learn(self, state, action, reward, next_state, weight):
        '''
        Update based on given data.
        '''
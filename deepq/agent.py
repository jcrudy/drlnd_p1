from collections import namedtuple
import numpy as np
from .policy.epsilon_greedy import EpsilonGreedyPolicy
from tqdm import tqdm
from .base import numpify, rolling_mean
from infinity import inf
from matplotlib import pyplot as plt
import pickle
import pandas

Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", 
                                     "next_state", "done"])

class Agent(object):
    def __init__(self, model, replay_buffer, training_policy, 
                 testing_policy=EpsilonGreedyPolicy(0, 0, 0), 
                 learn_every=4, batch_size=64):
        self.model = model
        self.replay_buffer = replay_buffer
        self.training_policy = training_policy
        self.testing_policy = testing_policy
        self.learn_every = learn_every
        self.batch_size = batch_size
        self.t = 0
        self.episodes_trained = 0
        self.train_scores = []
        self.test_scores = []
        self.train_episode_lengths = []
        self.test_episode_lengths = []
    
    def to_pickle(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as outfile:
            result = pickle.load(outfile)
        if not isinstance(result, cls):
            raise TypeError('Unpickled object is not correct type.')
        return result

    def plot_train_scores(self, episodes=inf, window=100):
        x, y = map(np.array, zip(*self.train_scores))
        rolling_y = rolling_mean(y, window)
        idx = x > (self.episodes_trained - episodes)
        x = x[idx]
        y = y[idx]
        rolling_y = rolling_y[idx]
        plt.plot(x, y, label='Training Episode Scores')
        plt.plot(x, rolling_y, label='Training Rolling Average Scores ({})'.format(window))
    
    def plot_test_scores(self, episodes=inf):
        x, y = map(np.array, zip(*self.test_scores))
        idx = x > (self.episodes_trained - episodes)
        x = x[idx]
        y = y[idx]
        df = pandas.DataFrame(dict(x=x, y=y))
        df = df.groupby('x').aggregate([np.mean, np.std])
        plt.errorbar(df.index, df[('y', 'mean')], yerr=2 * df[('y', 'std')], 
                     fmt='r.', ecolor='r', label='Test Scores', zorder=10)
        
    def learn(self):
        sample_indices, sample_probs = self.replay_buffer.sample_indices(self.batch_size)
        sample = self.replay_buffer[sample_indices]
        state, action, reward, next_state, done = map(np.array, zip(*sample))
        self.model.learn(state, action, reward, next_state, done, sample_probs)
    
    def train(self, environment, num_episodes=inf, validate_every=None, validation_size=10,
              save_every=None, save_path=None):
        '''
        Train for num_episodes episodes and return the episode scores.
        '''
        scores = []
        
        with tqdm(total=num_episodes) as t:
            for episode in range(num_episodes):
                # Run validation if appropriate
                if (
                    (validate_every is not None) and 
                    (episode % validate_every == 0)
                    ):
                    self.test(environment, validation_size)
                
                # Train for one episode
                episode_score = self.train_episode(environment)
                
                # Record score from training episode
                scores.append(episode_score)
                
                # Update the progress bar
                t.update(1)
                t.set_description('Last Episode Reward: {}'.format(episode_score))
                
                # Save progress if appropriate
                if (
                    (save_every is not None) and
                    (episode % save_every == 0)
                    ):
                    path = save_path.format(num_episodes)
                    self.to_pickle(path)
                
        return scores
    
    def test(self, environment, num_episodes):
        '''
        Run for num_episodes episodes under the testing policy and return 
        the episode scores.
        '''
        scores = []
        with tqdm(total=num_episodes) as t:
            for _ in range(num_episodes):
                scores.append(self.test_episode(environment))
                t.update(1)
        return scores
    
    def train_episode(self, environment):
        '''
        Execute an episode of training and return the total reward.
        '''
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        action_count = 0
        while not done:
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.training_policy.choose(numpify(values))
            next_state, reward, done = environment.step(action)
            episode_score += reward
            self.t += 1
            
            # Store experience for later learning.
            experience = Experience(state, action, reward, next_state, done)
            self.replay_buffer.append(experience)
            
            # Update state for next iteration
            state = next_state
            
            # Learn from recorded experiences.
            if self.t % self.learn_every == 0:
                self.learn()
                
            action_count += 1
        
        self.train_episode_lengths.append(action_count)
        self.train_scores.append((self.episodes_trained, episode_score))
        self.episodes_trained += 1
        return episode_score
    
    def test_episode(self, environment):
        '''
        Execute an episode under the testing policy.
        '''
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        action_count = 0
        while not done:
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.testing_policy.choose(numpify(values))
            next_state, reward, done = environment.step(action)
            episode_score += reward
            
            # Update state for next iteration
            state = next_state
            
            action_count += 1
        
        self.test_episode_lengths.append(action_count)
        self.test_scores.append((self.episodes_trained, episode_score))
        return episode_score
    

    
        
# class FixedSizeReplayBuffer(ReplayBuffer):
#     def __init__(self, action_size, buffer_size):
#         super().__init__(action_size=action_size)
#         self.buffer_size = buffer_size
#         self.buffer = deque(maxlen=self.buffer_size)
# 
#     def append(self, experience):
#         self.buffer.append(experience)
#     
#     def __getitem__(self, indices):
#         return tuple(self.buffer[idx] for idx in indices)



# class PrioritySamplingReplayBuffer(ReplayBuffer):
#     def __init__(self, action_size, buffer_size, ):

# class PrioritySamplingReplayBuffer(ReplayBuffer):
#     def __init__(self, ):

# class BasicReplayBuffer(FixedSizeReplayBuffer, UniformSamplingReplayBuffer):
#     pass
    
    









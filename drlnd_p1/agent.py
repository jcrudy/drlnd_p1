from collections import namedtuple
import numpy as np
from .policy.epsilon_greedy import EpsilonGreedyPolicy

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
    
#     def choose_greedy_action(self, values):
#         return np.random.choice(np.argmax(values))
#     
#     def choose_random_action(self, values):
#         return np.random.choice(len(values))
#     
#     def choose_epsilon_greedy_action(self, values):
#         if random.random() < self.epsilon:
            
    
    def learn(self):
        sample_indices, sample_probs = self.replay_buffer.sample(self.batch_size)
        sample = self.replay_buffer[sample_indices]
        state, action, reward, next_state = map(np.array, zip(*sample))
        self.model.learn(state, action, reward, next_state, sample_probs)
    
    def train(self, environment, num_episodes):
        '''
        Train for num_episodes episodes and return the episode scores.
        '''
        scores = []
        for _ in range(num_episodes):
            scores.append(self.train_episode(environment))
        return scores
    
    def test(self, environment, num_episodes):
        '''
        Run for num_episodes episodes and return the episode scores.
        '''
        scores = []
        for _ in range(num_episodes):
            scores.append(self.test_episode(environment))
        return scores
    
    def train_episode(self, environment):
        '''
        Execute an episode of training and return the total reward.
        '''
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        while not done:
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.training_policy.choose(values)
            next_state, reward, done = environment.step(action)
            episode_score += reward
            self.t += 1
            
            # Store experience for later learning.
            experience = Experience(state, action, reward, next_state, done)
            self.buffer.append(experience)
            
            # Update state for next iteration
            state = next_state
            
            # Learn from recorded experiences.
            if self.t % self.learn_every == 0:
                self.learn()
        
        return episode_score
    
    def test_episode(self, environment):
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=False)
        done = False
        while not done:
            # Choose and take an action.
            action = self.model.choose_action(state, train=False)
            next_state, reward, done = environment.step(action)
            episode_score += reward
            
            # Update state for next iteration
            state = next_state
        
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
    
    









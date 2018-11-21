from deepq.environment.unity_adapter import BananaEnvironment
from deepq.model.fixed_q_target import FixedQTargetModel
from deepq.agent import Agent
from deepq.buffer.uniform_sampling import UniformSamplingReplayBuffer
from deepq.buffer.prioritized_replay import PrioritizedReplayBuffer
from deepq.policy.epsilon_greedy import EpsilonGreedyPolicy
from deepq.network.base import Network
import torch.nn as nn
import torch.nn.functional as F
from toolz import identity

def main(args):
    # Get command line arguments
    num_episodes = args.n
    model_path = args.m
    validate_every = args.v
    validation_episodes = args.e
    save_every = args.s
    save_filename = args.f
    
    # Create the training environment
    environment = BananaEnvironment()
    
    # Attempt to load the agent.  Create a new agent if loading fails.
    try:
        agent = Agent.from_pickle(model_path)
    except FileNotFoundError:
        hidden_size = 100
        buffer_size = 100
        network = Network(state_size=environment.state_size, 
                          n_actions=environment.n_actions,
                          layers=(nn.Linear(environment.state_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, environment.n_actions)),
                          activations=(F.relu, F.relu, identity))
        model = FixedQTargetModel(network)
        buffer = PrioritizedReplayBuffer(buffer_size)
        training_policy = EpsilonGreedyPolicy(1., .995, .05)
        agent = Agent(model=model, replay_buffer=buffer, 
                  training_policy=training_policy)
    
    # Train the agent
    agent.train(environment, num_episodes, validate_every=validate_every,
                validation_size=validation_episodes, save_every=save_every,
                save_path=save_filename)
    
    # Save trained agent to disk
    agent.to_pickle(model_path)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a deep Q network')
    parser.add_argument('-m', metavar='<model_path>', 
                        help='The path of the model file.', required=True)
    parser.add_argument('-n', metavar='<num_episodes>', type=int,
                        help='The number of episodes for which to train.',
                        default=1000)
    parser.add_argument('-v', metavar='<validate_every>', 
                        help='The number of episodes between validations.',
                        default=100, type=int)
    parser.add_argument('-e', metavar='<validation_episodes>',
                        help='The number of episodes for which to validate during each validation',
                        default=100, type=int)
    parser.add_argument('-s', metavar='<save_every>',
                        help='The number of episodes between saves.', 
                        default=None, type=int)
    parser.add_argument('-f', metavar='<save_filename>',
                        help='Filename for saving intermediate results.  Will be formatted with number of episodes before saving.',
                        default=None)
    
    args = parser.parse_args()
    
    main(args)
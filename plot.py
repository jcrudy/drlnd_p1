from deepq.agent import Agent
from matplotlib import pyplot as plt
from infinity import inf

def main(args):
    # Get command line arguments
    num_episodes = args.e
    model_path = args.m
    
    # Load agent from disk
    agent = Agent.from_pickle(model_path)
    
    # Make a plot
    agent.plot_train_scores(episodes=num_episodes)
    agent.plot_test_scores(episodes=num_episodes)
    
    plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a deep Q network')
    parser.add_argument('-m', metavar='<model_path>', 
                        help='The path of the model file.',
                        default='banana_agent.pkl')
    parser.add_argument('-e', metavar='<num_episodes>', type=int,
                        help='The number of episodes to plot.',
                        default=inf)
    args = parser.parse_args()
    
    main(args)
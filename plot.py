from deepq.agent import Agent
from matplotlib import pyplot as plt
from infinity import inf

def main(args):
    # Get command line arguments
    num_episodes = args.e
    model_path = args.m
    outfilename = args.f
    
    # Load agent from disk
    agent = Agent.from_pickle(model_path)
    
    # Make a plot
    agent.plot_train_scores(episodes=num_episodes)
    agent.plot_test_scores(episodes=num_episodes)
    l, r = plt.xlim()
    plt.hlines(13., l, r, colors='r', linestyles='dotted', zorder=11)
    plt.xlim(l, r)
    
    # Save to file
    if outfilename is not None:
        plt.savefig(outfilename)
    
    print('Agent had {} episodes of training.'.format(agent.episodes_trained))
    plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot the past rewards of a trained agent.')
    parser.add_argument('-m', metavar='<model_path>', 
                        help='The path of the model file.',
                        default='banana_agent.pkl')
    parser.add_argument('-e', metavar='<num_episodes>', type=int,
                        help='The number of episodes to plot.',
                        default=inf)
    parser.add_argument('-f', metavar='<filename>',
                        help='File to which to save the plot.',
                        default=None)
    args = parser.parse_args()
    
    main(args)
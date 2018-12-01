from deepq.agent import Agent

def main(args):
    # Get command line arguments
    model_path = args.m
    outfilename = args.f
    
    # Load agent from disk
    agent = Agent.from_pickle(model_path)
    
    # Save to file
    if outfilename is not None:
        agent.save_weights(outfilename)
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Save the weights of a pickled agent.')
    parser.add_argument('-m', metavar='<model_path>', 
                        help='The path of the model file.',
                        default='banana_agent.pkl')
    parser.add_argument('-f', metavar='<filename>',
                        help='File to which to save the weights.',
                        default='banana_checkpoint.pth')
    args = parser.parse_args()
    
    main(args)
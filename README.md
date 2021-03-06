Navigation Project
==================

This repository is a submission for the navigation project of Udacity's deep reinforcement learning course.


Project Details
---------------

The assignment is to solve the Unity bananas environment, which consists of a cart that drives around a field collecting bananas by running them over.  The field has both yellow and blue bananas.  Yellow bananas yield a reward of 1 while blue bananas yield a reward of -1.  The environment is considered solved when an agent achieves an average return of 13 or more over 100 episodes of 300 actions each.  Actions are left and right turns and moves forward and backward.  The state is a 37 dimensional vector.  I solved the environment with an implementation of deep Q-learning.


Getting Started
---------------

This repository is designed to be easy to set up on all supported platforms, but was only tested on OSX.  Make sure you are using an anaconda python distribution, clone the repository, then create the project environment by running:

    conda env create

from the repositories root directory.  This will create the drlnd_p1 environment, which can then be
activated by:

    source activate drlnd_p1
    
The drlnd_p1 environment contains the following packages installed via conda:

    - nose=1.3.7
    - numpy=1.14.5
    - matplotlib=3.0.2
    - pytorch=0.4.0
    - protobuf=3.5.2
    - toolz=0.9.0
    - multipledispatch=0.6.0
    - infinity=1.4
    - six=1.11.0

Packages `gym` and `mlagents` are also installed via pip from their respective github repositories.  See environment.yml for details. 
    

Next, you can run the project's limited unit tests by running:

    nosetests

The project should download and extract the bananas environment automatically.  If it fails, please place the downloaded and extracted bananas environment file in deepq/environment/resources and try the tests again.  It may be necessary to fix the permissions of the extracted file.


Instructions
------------

Once the above environment has been installed and activated and the tests pass, a new agent can be created and trained by running train.py:

    python train.py

The train.py script will train a new agent and, by default, save that agent as banana_agent.pkl.  A plot of episode rewards will be displayed during training, and training will stop as soon as an average reward of 13 or more over 100 episodes has been achieved.
 
    

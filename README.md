# Team Name: Balalayka
# Team Members:
- Chris Daniels
- Dmitri Koltsov
# pacman-ctf
## Python3 version of UC Berkeley's CS 188 Pacman Capture the Flag project

### Original Licensing Agreement (which also extends to this version)
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

### This version attribution
This version (cshelton/pacman-ctf github repo) was modified by Christian
Shelton (cshelton@cs.ucr.edu) on June 23, 2020 to run under Python 3.


## Getting Started with Comrades
### Testing
To run Comrades against other agents, the Python file `myTeam_RL_Agent_1.py` is to be used. 
Since `myTeam_RL_Agent_1.py` has been trained to do well, the models that were trained are required to be
in the same directory as this Python file. The models are called `model_North.pth` and `model_South.pth`. 

<br>The commands to run Comrades against an opponent:
<br>- As red team: `python capture.py --red=myTeam_RL_Agent_1`
<br>- As blue team: `python capture.py --blue=myTeam_RL_Agent_1`
<br>
Any extra flags can be added to run on different layouts, and against any other customer agent.

### Training
The file used for training is `myTeam_RL_dueling.py`. To train the agent from scratch, make sure to delete the current models.
Training from scratch will create new models that can be used by `myTeam_RL_Agent_1.py`. 
Training can also use the current models and update them. To get a better idea of how
Comrades learn, training from scratch displays differences in win rates properly. 

<br>The commands to train Comrades against an opponent:
<br>- As red team: `python capture.py --red=myTeam_RL_dueling -n N -q`
<br>- As blue team: `python capture.py --blue=myTeam_RL_dueling -n N -q`
<br>Where `N` is how many games you want to play.

## Design
### Reinforcement Learning
Many games, especially Atari games, are the focus for reinforcement learning. Pacman being an Atari game sparked the 
interest in using this form of learning. Our agent, Comrades uses reinforcement learning to navigate the mazes, and 
maneuver their way around enemies. Reinforcement learning is perfect for allowing your agent to learn on its own by rewards and penalties.

### PyTorch
In order to make it easier on us, [PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) was used for reinforcement learning.
A Dueling Deep Q Network was used to get a Q value. A total of 37 features are fed into the network, with 5 outputs in the final layer. 

### Final Report
To get the full details of this project, please find `report.pdf` in our base directory.
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


## Agent 0

### Strategy:
For our agents we use value function to determine best next action.
- if maze-distance to the food decreases, the value (of the value function) increases. Otherwise it decreases.
- if maze-distance to the enemy decreases, the value (of the value function) increases. Otherwise it decreases.

Also we take into account positions of the agents (on what side of the board we are), maze configuration (are we are approaching a dead end), “scary” agents etc. All those criteria can modify or reverse value function components.

### How:
For calculating value of the distance change to the enemy we used formula: `log2(next distance / current distance)`.
For the value of the distance change to the food we used a constant.
We used a “stomach size” variable for our pacmans. So they should go back to the home side to deposit the food if they ate enough. Stomach size depends on current food amount on the board. Mostly it is 1/3 of the total food on the board.
When the “stomach” is full, our agent stops seeking food and tries to find the path to the home side.

### Contributions:
- Dmitri:
  - Wrote agents `Agent_North` and `Agent_South` that consistently beat the baseline team.
  - Came up with the strategy
  - Committed and pushed to github

- Chris
  - Created and updated agent0 branch to hold current agents
  - Wrote an agent `UpFucker` that will not be used currently
  - Modified README

### Performance of Agent0:
First our agent didn’t work very well, so we had to modify the “weights” of our values to make them more or less significant depending on the situation.
Finally, our agents easily beat the baseline team a majority of the games run.

### Lessons learned
We tried to compete against the agents of other groups, and almost every time agents (from both sides) stack in repetitive situations.
Opposing agents will meet up, and constantly "dance" in front of each other. Once one of our agents meets an opposing agent, it will take what it considers the best move. The opposing agent then makes that same move, causing our agent to revert back to the previous situation. This continues until one agent is eaten.
This is because algorithm is sort of “greedy” and always tries to move to the best state.
We are looking forward to create learning algorithm (or algorithm which can look several moves ahead), so we can solve this problem. 

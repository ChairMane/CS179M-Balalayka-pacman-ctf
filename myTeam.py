# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'DummyAgent', second = 'DummyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        global best_action
        actions = gameState.getLegalActions(self.index)
        if len(actions) is 1:
            return actions[0]

        '''
        You should change this in your own agent.
        '''
        value = -10
        for action in actions:
            new_state = self.getSuccessor(gameState, action)
            new_value = self.action_value(gameState, new_state)
            if new_value > value:
                best_action = action
                value = new_value

        return best_action



    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    
    def action_value(self, gameState, successor):
        value = 0
        current_food_positions = self.getFood(gameState).asList()
        next_food_positions = self.getFood(successor).asList()

        my_current_position = gameState.getAgentState(self.index).getPosition()
        my_next_position = successor.getAgentState(self.index).getPosition()

        if len(current_food_positions) is not len(next_food_positions):
            value += 0.8
        else:
            current_food_distance = min([self.getMazeDistance(my_current_position, food) for food in current_food_positions])
            next_food_distance = min([self.getMazeDistance(my_next_position, food) for food in next_food_positions])
            value += 1 / next_food_distance - 1 / current_food_distance

        enemy_indices = self.getOpponents(gameState)
        #tempo_indices = self.getOpponents(successor)
        enemy_positions_value = 0
        closest_enemy_distance = 1000
        flag = False
        for index in enemy_indices:
            current_pos = gameState.getAgentPosition(index)
            if current_pos:
                current_distance = self.getMazeDistance(my_current_position, current_pos)
                if current_distance < closest_enemy_distance:
                    closest_enemy_distance = current_distance
                enemy_positions_value += 1 / current_distance
            next_pos = successor.getAgentPosition(index)
            if next_pos:
                next_distance = self.getMazeDistance(my_next_position, next_pos)
                enemy_positions_value -= 1 / next_distance
            else:
                flag = True

        if my_next_position[0] > 14:
            value += enemy_positions_value
        else:
            value -= enemy_positions_value
            if flag and closest_enemy_distance is 1:
                value += 1
        if len(successor.getLegalActions(self.index)) is 1 and closest_enemy_distance < 4:
            value -= 1
        return value






        
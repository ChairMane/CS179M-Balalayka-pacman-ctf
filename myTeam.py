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
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='Agent_North', second='Agent_South'):
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

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.food_inside = 0
        self.flag_eat_mode = True
        #self.drop_positions = self.get_drop_positions()
        self.current_food_positions = []

    def get_drop_positions(self, gameState):
        positions = []
        for y in range(1, 15):
            if not gameState.hasWall(15, y):
                positions.append((15, y))
        return positions

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
        # global best_action
        #my_dist = gameState.getAgentDistances()

        global best_action
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        self.my_current_position = gameState.getAgentState(self.index).getPosition()

        if self.flag_eat_mode:
            self.current_food_positions = self.getFood(gameState).asList()
            self.current_food_amount = len(self.current_food_positions)
            self.my_food_positions = self.get_my_food_positions(gameState)
            if len(self.my_food_positions) > 0:
                self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])

        self.enemy_indices = self.getOpponents(gameState)
        self.current_enemy_positions = []
        self.current_enemy_distanses_positions = []
        for index in self.enemy_indices:
            pos = gameState.getAgentPosition(index)
            if pos:
                self.current_enemy_distanses_positions.append((self.getMazeDistance(self.my_current_position, pos), pos))

        if self.my_current_position[0] < 16:
            self.food_inside = 0
        pacman_stomach_size = int(self.current_food_amount / 3)
        if self.current_food_amount < 3 or self.food_inside > pacman_stomach_size:
            self.flag_eat_mode = False
        else:
            self.flag_eat_mode = True

        if not self.flag_eat_mode:
            self.drop_positions = self.get_drop_positions(gameState)
            self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])


        action_value = -10
        flag_food_eaten = False
        for action in actions:
            if action is 'Stop':
                new_action_value = -0.3
                tempo_flag = False
            else:
                new_action_value, tempo_flag = self.action_value(gameState, action)
            if new_action_value > action_value:
                best_action = action
                action_value = new_action_value
                flag_food_eaten = tempo_flag

        if flag_food_eaten:
            self.food_inside += 1

        return best_action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def get_my_food_positions(self, gameState):
        return self.current_food_positions

    def enemy_distance_value(self, dist_current, dist_next):
        return math.log2((dist_next + 1) / (dist_current + 1))

    def object_distance_value(self, dist_current, dist_next):
        return 0.003 * (dist_next**3 - dist_current**3)

    def action_value(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        value = 0

        flag_food_taken = False

        my_next_position = successor.getAgentState(self.index).getPosition()
        object_distance_value = 0.2
        if self.flag_eat_mode:
            next_food_distance = min([self.getMazeDistance(my_next_position, food) for food in self.my_food_positions])
            if next_food_distance < self.my_food_distance:
                value += object_distance_value
                if next_food_distance is 0:
                    flag_food_taken = True
                #     next_food_distance = 0.5
            elif next_food_distance > self.my_food_distance:
                value -= object_distance_value
            # value += 1 / next_food_distance - 1 / self.my_food_distance
        elif self.food_inside > 0:
            next_drop_distance = min([self.getMazeDistance(my_next_position, drop) for drop in self.drop_positions])
            if next_drop_distance < self.current_drop_distance:
                value += object_distance_value
                if next_drop_distance is 0:
                    self.food_inside = 0
                    self.flag_eat_mode = True
                    return 5, False
            elif next_drop_distance > self.current_drop_distance:
                value -= object_distance_value
            # value += 1 / next_drop_distance
            # value -= 1 / self.current_drop_distance

        # if my_next_position is (1, 1):
        #     value -= 2

        enemy_positions_value = 0
        closest_enemy_distance = 1000
        current_enemy_distance = 1000
        for item in self.current_enemy_distanses_positions:
            current_enemy_distance = item[0]
            current_enemy_position = item[1]
            next_enemy_distance = self.getMazeDistance(my_next_position, current_enemy_position)
            enemy_position_change = self.enemy_distance_value(current_enemy_distance, next_enemy_distance)
            if my_next_position[0] > 15:
                if current_enemy_position[0] < 15:
                    enemy_positions_value -= enemy_position_change
                else:
                    enemy_positions_value += enemy_position_change
            else:
                enemy_positions_value -= enemy_position_change

            # if next_enemy_distance is 0:
            #     next_enemy_distance = 0.5
            #     if current_enemy_position[0] < 16:
            #         value += 1
            if current_enemy_distance < closest_enemy_distance:
                closest_enemy_distance = current_enemy_distance
            # enemy_positions_value += 1 / current_enemy_distance - 1 / next_enemy_distance
        value += enemy_positions_value
        if my_next_position[0] > 15:
            if len(successor.getLegalActions(self.index)) is 2:
                value -= 0.2
                if closest_enemy_distance < 4:
                    value -= 0.2

        return value, flag_food_taken

class Agent_North(DummyAgent):
    def get_my_food_positions(self, gameState):
        n = len(self.current_food_positions)
        sum = 0
        for item in self.current_food_positions:
            sum += item[1]
        avg = sum / n
        result = []
        for pos in self.current_food_positions:
            if pos[1] > avg:
                result.append(pos)
        return result

class Agent_South(DummyAgent):
    def get_my_food_positions(self, gameState):
        n = len(self.current_food_positions)
        sum = 0
        for item in self.current_food_positions:
            sum += item[1]
        avg = sum / n
        result = []
        for pos in self.current_food_positions:
            if pos[1] <= avg:
                result.append(pos)
        return result
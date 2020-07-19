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
from game import Grid
import game
import math
import time
import pandas as pd
import numpy as np
import functools
import operator
from sklearn.neural_network import MLPClassifier
import joblib


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
        self.field_width = gameState.data.layout.width
        self.field_height = gameState.data.layout.height
        self.field_mid_width = int((self.field_width - 2) / 2)
        self.field_mid_height = int((self.field_height - 2) / 2)
        self.my_indices, self.enemy_indices = self.get_indices(gameState)
        self.food_inside = 0
        self.current_food_positions = []
        self.score = 0

        self.data_grid_radius = 5
        self.features_groups = 9
        self.qualities = 9
        self.data_set_current = []

        #self.wh, self.bh, self.wo, self.bo = self.read_weights()
        self.my_model = self.read_model()

    def get_indices(self, gameState):
        if self.red:
            return gameState.getRedTeamIndices(), gameState.getBlueTeamIndices()
        else:
            return gameState.getBlueTeamIndices(), gameState.getRedTeamIndices()

    def create_state_data(self, gameState):
        base_field = np.zeros([self.field_height, self.field_width])
        for j in range(self.field_height):
            for i in range(self.field_width):
                if gameState.hasWall(i, j):
                    base_field[j,i] = 1

        x = int(self.my_current_position[0])
        y = int(self.my_current_position[1])
        rad = self.data_grid_radius
        x_0 = x - rad
        y_0 = y - rad
        x_1 = x + rad
        y_1 = y + rad
        n = self.data_grid_radius * 2 + 1
        n_sqr = n * n
        grid_positions = np.zeros([self.features_groups, n_sqr], dtype=int)
        grid_qualities = np.zeros(self.qualities, dtype=int)
        for j in range(n):
            y_current = y_0 + j
            if y_current < 0:
                continue
            if y_current >= self.field_height:
                break
            for i in range(n):
                x_current = x_0 + i
                if x_current < 0:
                    continue
                if x_current >= self.field_width:
                    break
                # what inside the grid grid_positions[0]
                grid_positions[0, j * n + i] = 1
                # walls grid_positions[1]
                if gameState.hasWall(x_current, y_current):
                    grid_positions[1, j * n + i] = 1
        # food for me grid_positions[2]
        for pos in self.current_food_positions:
            (x_t, y_t) = pos
            if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                grid_positions[2, (y_t - y_0) * n + x_t - x_0] = 1
        # food for enemy grid_positions[3]
        for pos in self.enemy_food_positions:
            (x_t, y_t) = pos
            if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                grid_positions[3, (y_t - y_0) * n + x_t - x_0] = 1
        # power cell for me grid_positions[4]
        for pos in self.capsules_for_me:
            (x_t, y_t) = pos
            if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                grid_positions[4, (y_t - y_0) * n + x_t - x_0] = 1
        # power cell for enemy grid_positions[5]
        for pos in self.capsules_for_enemy:
            (x_t, y_t) = pos
            if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                grid_positions[5, (y_t - y_0) * n + x_t - x_0] = 1
        # friendly agent position grid_positions[6]
        # friendly scary timer grid_qualities[0] (self) and grid_qualities[1] (friend)
        for ind in self.my_indices:
            pos = gameState.getAgentPosition(ind)
            (x_t, y_t) = pos
            if pos == self.my_current_position:
                grid_qualities[0] = gameState.getAgentState(ind).scaredTimer
                # relative x of the agent
                grid_qualities[5] = (x_t - self.field_mid_width) / self.field_width
                # relative y of the agent
                grid_qualities[6] = (y_t - self.field_mid_height) / self.field_height
            else:
                grid_qualities[1] = gameState.getAgentState(ind).scaredTimer
                # relative x of the friendly agent
                grid_qualities[7] = (x_t - self.field_mid_width) / self.field_width
                # relative y of the friendly agent
                grid_qualities[8] = (y_t - self.field_mid_height) / self.field_height
                if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                    grid_positions[6, (y_t - y_0) * n + x_t - x_0] = 1
        # enemy positions grid_positions[7] and grid_positions[8]
        # enemy scary timer grid_qualities[2] and grid_qualities[3]
        for i, ind in enumerate(self.enemy_indices):
            pos = gameState.getAgentPosition(ind)
            grid_qualities[2 + i] = gameState.getAgentState(ind).scaredTimer
            if pos:
                (x_t, y_t) = pos
                if x_t >= x_0 and x_t <= x_1 and y_t >= y_0 and y_t <= y_1:
                    grid_positions[7 + i, (y_t - y_0) * n + x_t - x_0] = 1
        # food inside
        grid_qualities[4] = self.food_inside

        return np.concatenate((base_field.ravel(), grid_positions.ravel(), grid_qualities))

    def add_move(self, act):
        move = np.zeros(5, dtype=int)
        def stop():
            move[0] = 1
        def north():
            move[1] = 1
        def east():
            move[2] = 1
        def south():
            move[3] = 1
        def west():
            move[4] = 1
        switcher = {
            'Stop': stop,
            'North': north,
            'East': east,
            'South': south,
            'West': west
        }
        switcher[act]()
        return move

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tang_h(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0)

    def normalize(self, features):
        features_number = len(features)
        m = np.mean(features)
        sd = np.sqrt(np.sum((features - m) ** 2) / (features_number - 1))
        return (features - m) / sd

    def create_moves(self, wh, bh, wo, bo, features):
        features_z_norm = self.normalize(features)
        sh = np.dot(features_z_norm, wh) + bh
        ah = self.sigmoid(sh)
        so = np.dot(ah, wo) + bo
        return self.softmax(so)

    def moves_to_act(self, moves, actions):
        indices = moves.argsort()[::-1]
        action_dictionary = {0: 'Stop', 1: 'North', 2: 'East', 3: 'South', 4: 'West'}
        for index in indices:
            best_action = action_dictionary[index]
            if best_action in actions:
                return best_action


    def food_eaten_flag(self, gameState, best_action):
        flag = False
        successor = self.getSuccessor(gameState, best_action)
        if self.red:
            food = successor.getBlueFood().asList()
        else:
            food = successor.getRedFood().asList()
        if self.current_food_amount == len(food) + 1:
            flag = True
        return flag

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        #time.sleep(0.06)

        # machine learning data collection

        #self.collecting_data(gameState)

        # end data collection

        global best_action

        '''
        You should change this in your own agent.
        '''

        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.is_home = self.at_home(self.my_current_position, 0)
        if self.is_home:
            self.food_inside = 0

        blue_food = gameState.getBlueFood().asList()
        red_food = gameState.getRedFood().asList()
        blue_capsules = gameState.getBlueCapsules()
        red_capsules = gameState.getRedCapsules()
        if self.red:
            self.current_food_positions = blue_food
            self.enemy_food_positions = red_food
            self.capsules_for_me = blue_capsules
            self.capsules_for_enemy = red_capsules
        else:
            self.current_food_positions = red_food
            self.enemy_food_positions = blue_food
            self.capsules_for_me = red_capsules
            self.capsules_for_enemy = blue_capsules

        self.current_food_positions.sort(key = lambda x: x[1])
        self.current_food_amount = len(self.current_food_positions)
        self.my_food_positions = self.current_food_positions
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])

        state_data = self.create_state_data(gameState)
        features = np.asarray(state_data)
        #moves = self.create_moves(self.wh, self.bh, self.wo, self.bo, features)
        moves = self.my_model.predict_proba([features])[0]
        actions = gameState.getLegalActions(self.index)
        best_action = self.moves_to_act(moves, actions)

        flag_food_eaten = self.food_eaten_flag(gameState, best_action)

        if flag_food_eaten:
            self.food_inside += 1

        #self.data_set_current.append(np.concatenate((state_data, self.add_move(best_action))))

        return best_action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def at_home(self, my_pos, deep):
        if (self.red and my_pos[0] <= self.field_mid_width - deep) or (
                not self.red and my_pos[0] > self.field_mid_width + deep):
            return True
        return False

    def read_weights(self):
        return 0, 0, 0, 0

    def read_model(self):
        return None

    def collecting_data(self, gameState):
        return None

class Agent_North(DummyAgent):
    def get_my_food_positions(self):
        n = int(self.current_food_amount / 2)
        return self.current_food_positions[n:]

    def collecting_data(self, gameState):
        new_score = gameState.data.score
        if new_score != self.score:

            score_change = new_score - self.score
            self.score = new_score
            if score_change > 0:
                df = pd.DataFrame(self.data_set_current)
                df.to_csv('my_data_North_ML.csv', mode='a', header=False, index=False)
            del self.data_set_current[:]

    def read_weights(self):
        wh = np.load('wh_North.npy')
        bh = np.load('bh_North.npy')
        wo = np.load('wo_North.npy')
        bo = np.load('bo_North.npy')
        return wh, bh, wo, bo

    def read_model(self):
        return joblib.load('model_North.sav')


class Agent_South(DummyAgent):
    def get_my_food_positions(self):
        n = int((self.current_food_amount + 1) / 2)
        return self.current_food_positions[:n]

    def collecting_data(self, gameState):
        new_score = gameState.data.score
        if new_score != self.score:

            score_change = new_score - self.score
            self.score = new_score
            if score_change > 0:
                df = pd.DataFrame(self.data_set_current)
                df.to_csv('my_data_South_ML.csv', mode='a', header=False, index=False)
            del self.data_set_current[:]

    def read_weights(self):
        wh = np.load('wh_South.npy')
        bh = np.load('bh_South.npy')
        wo = np.load('wo_South.npy')
        bo = np.load('bo_South.npy')
        return wh, bh, wo, bo

    def read_model(self):
        return joblib.load('model_South.sav')
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

#from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

import keras
from keras import models
#from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


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
        self.food_inside_prev = 0
        self.drop_positions = self.get_drop_positions(gameState)
        self.current_food_positions = []
        self.flag_food_eaten = False
        self.flag_food_eaten_prev = False
        self.flag_death = False

        self.data_grid_radius = 5
        self.features_groups = 9
        self.qualities = 9
        self.data_set_current = []
        self.data_value = np.empty(0)
        self.prev_state_value = 0
        self.flag_win = False

        self.my_model, self.my_scaler = self.read_model_scaler()

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
            if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                grid_positions[2, (y_t - y_0) * n + x_t - x_0] = 1
        # food for enemy grid_positions[3]
        for pos in self.enemy_food_positions:
            (x_t, y_t) = pos
            if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                grid_positions[3, (y_t - y_0) * n + x_t - x_0] = 1
        # power cell for me grid_positions[4]
        for pos in self.capsules_for_me:
            (x_t, y_t) = pos
            if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                grid_positions[4, (y_t - y_0) * n + x_t - x_0] = 1
        # power cell for enemy grid_positions[5]
        for pos in self.capsules_for_enemy:
            (x_t, y_t) = pos
            if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
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
                if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                    grid_positions[6, (y_t - y_0) * n + x_t - x_0] = 1
        # enemy positions grid_positions[7] and grid_positions[8]
        # enemy scary timer grid_qualities[2] and grid_qualities[3]
        for i, ind in enumerate(self.enemy_indices):
            pos = gameState.getAgentPosition(ind)
            grid_qualities[2 + i] = gameState.getAgentState(ind).scaredTimer
            if pos:
                (x_t, y_t) = pos
                if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
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

    def all_food_positions(self, gameState):
        blue_food = gameState.getBlueFood().asList()
        red_food = gameState.getRedFood().asList()
        blue_capsules = gameState.getBlueCapsules()
        red_capsules = gameState.getRedCapsules()
        if self.red:
            current_food_positions = blue_food
            enemy_food_positions = red_food
            capsules_for_me = blue_capsules
            capsules_for_enemy = red_capsules
        else:
            current_food_positions = red_food
            enemy_food_positions = blue_food
            capsules_for_me = red_capsules
            capsules_for_enemy = blue_capsules
        return current_food_positions, enemy_food_positions, capsules_for_me, capsules_for_enemy

    def state_action_value(self, gameState):
        s_value = 0
        s_value += 1500 / (0.15 * self.my_food_distance + 7)**3
        s_value -= -3 * math.tanh(0.2 * self.getMazeDistance(self.my_current_position, (1, 1)) - 1.2) + 3
        if self.food_inside > 0:
            s_value += -3 * math.tanh(0.5 * self.current_drop_distance / self.food_inside - 1) + 3

        enemy_dist_value = 0
        for ind in self.enemy_indices:
            pos = gameState.getAgentPosition(ind)
            if pos:
                if gameState.getAgentState(ind).scaredTimer > 3 and not self.at_home(pos, 0):
                    continue
                distance_value = -3 * math.tanh(0.13 * self.getMazeDistance(self.my_current_position, pos) - 0.8) + 3
                enemy_is_home = self.at_home(pos, 0)
                if gameState.getAgentState(self.index).scaredTimer > 0:
                    enemy_dist_value -= distance_value
                else:
                    if enemy_is_home:
                        if self.is_home:
                            enemy_dist_value += distance_value
                        else:
                            enemy_dist_value += distance_value
                    else:
                        if self.is_home:
                            enemy_dist_value += distance_value
                        else:
                            enemy_dist_value -= distance_value
        s_value += enemy_dist_value

        return s_value

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

    def q_func(self):
        n = len(self.data_value)
        if n > 0:
            reward = np.logspace(1, 5, num=6, base=3) / 100
            if self.flag_death:
                if n >= 6:
                    self.data_value[-6:] -= reward * 3
                else:
                    self.data_value -= reward[-n:] * 3
            else:
                if (self.food_inside == 0 and self.food_inside_prev > 2) or self.flag_win:
                    if n >= 6:
                        self.data_value[-6:] += reward
                    else:
                        self.data_value += reward[-n:]
                if self.flag_food_eaten_prev:
                    if n >= 6:
                        self.data_value[-6:] += reward
                    else:
                        self.data_value += reward[-n:]
            if n > 1:
                self.data_value[-2] += 0.6 * (self.data_value[-1] - self.data_value[-2])
            # How to check enemy's death????

    def read_model_scaler(self):
        return None, None

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        #time.sleep(0.06)


        global best_action

        '''
        You should change this in your own agent.
        '''

        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.is_home = self.at_home(self.my_current_position, 0)
        if self.is_home:
            self.food_inside = 0
        else:
            self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])

        self.current_food_positions, self.enemy_food_positions, self.capsules_for_me, self.capsules_for_enemy = self.all_food_positions(gameState)

        #self.current_food_positions.sort(key = lambda x: x[1])
        self.current_food_amount = len(self.current_food_positions)
        self.my_food_positions = self.current_food_positions
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])

        state_data = self.create_state_data(gameState)

        actions = gameState.getLegalActions(self.index)
        action_value = -10
        if random.random() > 0.7:
            best_action = random.choice(actions)
        else:
            for act in actions:
                features = np.concatenate((state_data, self.add_move(act)))
                features = self.my_scaler.transform(features.reshape(1, -1))
                value = self.my_model.predict(features)
                if value > action_value:
                    best_action = act
                    action_value = value

        self.flag_food_eaten = self.food_eaten_flag(gameState, best_action)
        if self.flag_food_eaten:
            self.food_inside += 1

        self.flag_death = False
        if self.my_current_position == (1, 1):
            self.flag_death = True

        #self.data_set_current.append(np.concatenate((state_data, self.add_move(best_action))))
        #self.q_func()
        #self.data_value = np.concatenate((self.data_value, [self.state_action_value(gameState)]))

        #self.flag_food_eaten_prev = self.flag_food_eaten

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

    def get_drop_positions(self, gameState):
        positions = []
        x = self.field_mid_width
        if not self.red:
            x += 1
        h = int(self.field_mid_height * 2 + 1)
        for y in range(1, h):
            if not gameState.hasWall(x, y):
                positions.append((x, y))
        return positions


class Agent_North(DummyAgent):
    def get_my_food_positions(self):
        n = int(self.current_food_amount / 2)
        return self.current_food_positions[n:]

    def read_model_scaler(self):
        return models.load_model('model_North.hdf5'), joblib.load('scaler_North.sav')


class Agent_South(DummyAgent):
    def get_my_food_positions(self):
        n = int((self.current_food_amount + 1) / 2)
        return self.current_food_positions[:n]

    def read_model_scaler(self):
        return models.load_model('model_South.hdf5'), joblib.load('scaler_South.sav')
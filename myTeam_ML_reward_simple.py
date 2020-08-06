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

from sklearn.preprocessing import Normalizer
#from sklearn.neural_network import MLPRegressor
#import joblib

import keras
from keras.models import load_model
#import torch


#################
# Team creation #
#################
model_North = load_model('model_North.h5')
model_South = load_model('model_South.h5')


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
        self.flag_food_eaten = False # if pellet consumed by agent
        self.flag_food_eaten_prev = False # if pellet consumed in previous step
        self.flag_death = False # if agent got eaten

        self.data_grid_radius = 5
        self.features_groups = 9
        self.qualities = 13
        self.data_set_current = []

        self.epsilon = .7 # exploration rate
        self.gamma = 0.99 # gamma for discounted reward
        self.penalty = 0.1 # penalty for each turn

        self.rewards_values = np.empty(0) # reward for each step
        self.flag_win = False # if game won
        self.flag_lose = False # if game lost

        self.my_model = self.read_model()
        self.my_scaler = Normalizer()

    # return 2 arrays of our indices and enemy indices
    def get_indices(self, gameState):
        if self.red:
            return gameState.getRedTeamIndices(), gameState.getBlueTeamIndices()
        else:
            return gameState.getBlueTeamIndices(), gameState.getRedTeamIndices()

    # transform string action to the integer index
    def action_to_index(self, act):
        def stop():
            return 0
        def north():
            return 1
        def east():
            return 2
        def south():
            return 3
        def west():
            return 4
        switcher = {
            'Stop': stop,
            'North': north,
            'East': east,
            'South': south,
            'West': west
        }
        return switcher[act]()

    # features space of games state
    def create_state_data_simple(self, gameState):
        # food, drop predicted
        food_future_dist = np.full(5, 10 / (self.my_food_distance + 1))
        drop_future_dist = np.full(5, 10 / (self.current_drop_distance + 1))
        for action in self.actions:
            successor = self.getSuccessor(gameState, action)
            new_pos = successor.getAgentState(self.index).getPosition()
            i = self.action_to_index(action)
            if len(self.my_food_positions) == 0:
                food_dist = float('inf')
            else:
                food_dist = min([self.getMazeDistance(new_pos, food) for food in self.my_food_positions])
            drop_dist = min([self.getMazeDistance(new_pos, drop) for drop in self.drop_positions])
            food_future_dist[i] = 10 / (food_dist + 1)
            drop_future_dist[i] = 10 / (drop_dist + 1)

        grid_qualities = np.zeros(self.qualities, dtype=int)


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

        # enemy scary timer grid_qualities[2] and grid_qualities[3]
        enemy_future_dist = np.zeros((2, 5))
        for i, ind in enumerate(self.enemy_indices):
            pos = gameState.getAgentPosition(ind)
            grid_qualities[2 + i] = gameState.getAgentState(ind).scaredTimer
            if pos:
                if gameState.getAgentState(ind).scaredTimer > 3 and not self.at_home(pos, 0):
                    continue
                dist = 10 / (self.getMazeDistance(self.my_current_position, pos) + 1)
                grid_qualities[9 + i] = dist

                enemy_future_dist[i] = dist
                for action in self.actions:
                    successor = self.getSuccessor(gameState, action)
                    new_pos = successor.getAgentState(self.index).getPosition()
                    enemy_future_dist[i, self.action_to_index(action)] = 10 / (self.getMazeDistance(new_pos, pos) + 1)
        # food inside
        grid_qualities[4] = self.food_inside
        # amount of food for us
        grid_qualities[11] = len(self.current_food_positions)
        # amount of food for enemy
        grid_qualities[12] = len(self.enemy_food_positions)

        return np.concatenate((food_future_dist, drop_future_dist, enemy_future_dist.ravel(), grid_qualities))

    # return array like [0, 1, 0, 0, 0] where 1 indicate which action was taken
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

    # return arrays of positions of our food, enemy food, our capsules, enemy capsules
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

    # if action results in eating pellet
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

    # read model asd scaler from the file
    def read_model(self):
        return None

    # calculate and add reward for each turn to the reward array
    def add_reward(self):
        reward = -self.penalty
        if self.flag_death:
            reward -= 8
        else:
            if self.food_inside == 0:
                reward += self.food_inside_prev
            if self.flag_food_eaten_prev:
                reward += 1
            if  self.flag_win:
                reward += 15
            if self.flag_lose:
                reward -= 15
        self.rewards_values = np.concatenate((self.rewards_values, [reward]))

    # calculate returns for each step
    def calc_returns(self, rewards):
        n = rewards.shape[0]
        returns = np.zeros(n)
        for i in range(n):
            for j in range(n - i):
                returns[i] += rewards[i + j] * self.gamma**j
        return returns


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        #time.sleep(0.06)

        self.actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.is_home = self.at_home(self.my_current_position, 0)
        if self.is_home:
            self.food_inside = 0

        self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])

        self.current_food_positions, self.enemy_food_positions, self.capsules_for_me, self.capsules_for_enemy = self.all_food_positions(gameState)
        self.current_food_positions.sort(key=lambda x: x[1])
        self.current_food_amount = len(self.current_food_positions)
        self.my_food_positions = self.get_my_food_positions()
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])
        else:
            self.my_food_distance = float('inf')

        state_data = self.create_state_data_simple(gameState)

        action_value = float('-inf')
        best_action = 'Stop'
        if random.random() > self.epsilon:
            while best_action == 'Stop':
                best_action = random.choice(self.actions)
        else:
            for act in self.actions:
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

        self.data_set_current.append(np.concatenate((state_data, self.add_move(best_action))))
        self.add_reward()

        self.flag_food_eaten_prev = self.flag_food_eaten

        return best_action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    # check if position in our side of the board
    def at_home(self, my_pos, deep):
        if (self.red and my_pos[0] <= self.field_mid_width - deep) or (
                not self.red and my_pos[0] > self.field_mid_width + deep):
            return True
        return False

    # return array of all food-drop positions on the board
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

    def get_my_food_positions(self):
        return self.current_food_positions


class Agent_North(DummyAgent):
    def get_my_food_positions(self):
        n = int(self.current_food_amount / 2)
        return self.current_food_positions[n:]

    def read_model(self):
        return model_North

    def final(self, gameState):
        if gameState.data.score > 0:
            self.flag_win = True
        if gameState.data.score < 0:
            self.flag_lose = True
        self.add_reward()
        returns = self.calc_returns(self.rewards_values[1:])
        #print(returns)
        df = pd.DataFrame(np.column_stack((np.asarray(self.data_set_current), returns)))
        df.to_csv('my_data_North.csv', mode='a', header=False, index=False)


class Agent_South(DummyAgent):
    def get_my_food_positions(self):
        n = int((self.current_food_amount + 1) / 2)
        return self.current_food_positions[:n]

    def read_model(self):
        return model_South

    def final(self, gameState):
        if gameState.data.score > 0:
            self.flag_win = True
        if gameState.data.score < 0:
            self.flag_lose = True
        self.add_reward()
        returns = self.calc_returns(self.rewards_values[1:])
        #print(returns)
        df = pd.DataFrame(np.column_stack((np.asarray(self.data_set_current), returns)))
        df.to_csv('my_data_South.csv', mode='a', header=False, index=False)
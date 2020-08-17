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
#from game import Directions
#from game import Grid
#import game
import math
import time
#import pandas as pd
import numpy as np
#import functools
#import operator

from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

#import os.path
from os import path


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
        self.approaching_food_reward = 0
        self.prev_current_food_amount = self.get_enemy_food_amount(gameState)
        self.drop_positions = self.get_drop_positions(gameState)
        self.approaching_drop_reward = 0
        self.approaching_enemy_reward = 0
        self.flag_food_eaten = False # if pellet consumed by agent
        self.flag_death = False # if agent got eaten
        self.flag_enemy_around = False  # if enemy is around
        self.flag_enemy_death = False # if enemy got eaten

        self.data_grid_radius = 5
        self.features_groups = 9
        self.data_set_current = []
        self.data_actions = ['Stop']

        self.epsilon = 0.2 # exploration rate
        self.gamma = 0.99 # gamma for discounted reward
        self.penalty = 0 # penalty for each turn
        self.epochs = 100 # number of epochs for learning
        self.learning_step = 20 # update Q-target function

        self.flag_delay = False # slow game visualisation

        # -=REWARD modifiers=-
        self.reward_modifiers()

        # variables for functions and classes
        self.state_data = self.create_state_data_simple_v1
        self.add_reward = self.add_reward
        self.Duel_Q_Network = Duel_Q_Network_very_simple

        self.rewards_values = np.empty(0) # reward for each step
        self.flag_done = False # if game over

        self.online_Q_network, self.optimizer, self.my_history, self.total_epochs, self.num_games_played = self.load_model()

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

    # transform string actions array to integer index array
    def actions_to_indices(self, acts):
        result = []
        for act in acts:
            result.append(self.action_to_index(act))
        return result

    # transform index to string action
    def index_to_action(self, index):
        actions = ['Stop', 'North', 'East', 'South', 'West']
        return actions[index]

    # if action reversed
    def get_reverse(self, action):
        if action == 'North':
            return 'South'
        if action == 'East':
            return 'West'
        if action == 'South':
            return 'North'
        if action == 'West':
            return 'East'
        return action

    # return array like [0, 1, 0, 0, 0] where 1 indicate which action was taken
    def get_action_array(self, act):
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

    # return new position
    def action_to_pos(self, action, pos):
        new_pos = list(pos)
        def stop():
            return
        def north():
            new_pos[1] += 1
        def east():
            new_pos[0] += 1
        def south():
            new_pos[1] -= 1
        def west():
            new_pos[0] -= 1
        switcher = {
            'Stop': stop,
            'North': north,
            'East': east,
            'South': south,
            'West': west
        }
        switcher[action]()
        return tuple(new_pos)

    # features space of games state
    def create_state_data_v1(self, gameState):
        # food, drop predicted
        food_future_dist = np.full(5, 10 / (self.my_food_distance + 1))
        drop_future_dist = np.full(5, 10 / (self.current_drop_distance + 1))
        if not self.flag_done:
            for action in self.actions:
                successor = self.getSuccessor(gameState, action)
                new_pos = successor.getAgentState(self.index).getPosition()
                i = self.action_to_index(action)
                food_dist = min([self.getMazeDistance(new_pos, food) for food in self.my_food_positions])
                drop_dist = min([self.getMazeDistance(new_pos, drop) for drop in self.drop_positions])
                food_future_dist[i] = 10 / (food_dist + 1)
                drop_future_dist[i] = 10 / (drop_dist + 1)

        # data in square around agent of radius self.data_grid_radius
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
        enemy_future_dist = np.zeros((2, 5))
        for i, ind in enumerate(self.enemy_indices):
            pos = gameState.getAgentPosition(ind)
            grid_qualities[2 + i] = gameState.getAgentState(ind).scaredTimer
            if pos:
                if gameState.getAgentState(ind).scaredTimer > 3 and not self.at_home(pos, 0):
                    continue
                dist = 10 / (self.getMazeDistance(self.my_current_position, pos) + 1)
                (x_t, y_t) = pos
                if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                    grid_positions[7 + i, (y_t - y_0) * n + x_t - x_0] = 1
                enemy_future_dist[i] = dist
                if not self.flag_done:
                    for action in self.actions:
                        successor = self.getSuccessor(gameState, action)
                        new_pos = successor.getAgentState(self.index).getPosition()
                        enemy_future_dist[i, self.action_to_index(action)] = 10 / (self.getMazeDistance(new_pos, pos) + 1)
        # food inside
        grid_qualities[4] = self.food_inside
        # amount of food for us
        grid_qualities[9] = len(self.current_food_positions)
        # amount of food for enemy
        grid_qualities[10] = len(self.enemy_food_positions)
        # last action
        return np.concatenate((food_future_dist, drop_future_dist, enemy_future_dist.ravel(), grid_positions.ravel(), grid_qualities, self.get_action_array(self.data_actions[-1])))

    def create_state_data_simple(self, gameState):
        # food, drop predicted
        food_future_dist = np.full(5, 10 / (self.my_food_distance + 1))
        drop_future_dist = np.full(5, 10 / (self.current_drop_distance + 1))
        if not self.flag_done:
            for action in self.actions:
                successor = self.getSuccessor(gameState, action)
                new_pos = successor.getAgentState(self.index).getPosition()
                i = self.action_to_index(action)
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
                enemy_future_dist[i] = dist
                if not self.flag_done:
                    for action in self.actions:
                        successor = self.getSuccessor(gameState, action)
                        new_pos = successor.getAgentState(self.index).getPosition()
                        enemy_future_dist[i, self.action_to_index(action)] = 10 / (
                                    self.getMazeDistance(new_pos, pos) + 1)
        # food inside
        grid_qualities[4] = self.food_inside
        # amount of food for us
        grid_qualities[9] = len(self.current_food_positions)
        # amount of food for enemy
        grid_qualities[10] = len(self.enemy_food_positions)

        return np.concatenate((food_future_dist, drop_future_dist, enemy_future_dist.ravel(), grid_qualities))

    def create_state_data_simple_v1(self, gameState):
        # food, drop, capsule, enemy prediction per action: -1 for leave, 1 for approach
        food_future_dist = np.zeros(5)
        capsule_future_dist = np.zeros(5)
        drop_future_dist = np.zeros(5)
        enemy_future_dist = np.zeros(5)

        grid_qualities = np.zeros(7, dtype=int)

        if not self.flag_done:
            for action in self.actions:
                i = self.action_to_index(action)
                food_future_dist[i] = self.get_approaching_food_reward(gameState, action)
                capsule_future_dist[i] = self.get_approaching_capsule_reward(gameState, action)
                if self.food_inside >= self.stomach_size:
                    drop_future_dist[i] = self.get_approaching_drop_reward(gameState, action)

            # my scary timer grid_qualities[0] relative position grid_qualities[1]
            x_t = self.my_current_position[0]
            grid_qualities[0] = int(gameState.getAgentState(self.index).scaredTimer > 0)
            # relative x of the agent
            grid_qualities[1] = (x_t - self.field_mid_width) / self.field_width

            # enemy data
            if self.closest_enemy_data:
                pos, dist, timer = self.closest_enemy_data
                grid_qualities[2] = int(timer > 0)
                grid_qualities[3] = 5 / dist
                for action in self.actions:
                    i = self.action_to_index(action)
                    enemy_future_dist[i] = self.get_approaching_enemy_reward(gameState, action, pos, dist)


            grid_qualities[4] = self.current_food_amount
            grid_qualities[5] = 5 / self.my_capsule_distance
            grid_qualities[6] = self.food_inside
        # debugging
        # (food_future_dist, drop_future_dist, enemy_future_dist.ravel(), grid_qualities)

        return np.concatenate((food_future_dist, drop_future_dist, capsule_future_dist, enemy_future_dist,grid_qualities))

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

    # return initial amount of enemy food
    def get_enemy_food_amount(self, gameState):
        blue_food = gameState.getBlueFood().asList()
        red_food = gameState.getRedFood().asList()
        if self.red:
            return len(blue_food)
        else:
            return len(red_food)

    # return approaching food reward
    def get_approaching_food_reward(self, gameState, action):
        reward = 0
        successor = self.getSuccessor(gameState, action)
        new_pos = successor.getAgentState(self.index).getPosition()
        if len(self.my_food_positions) > 0:
            new_distance = min([self.getMazeDistance(new_pos, food) for food in self.my_food_positions])
            if new_distance > self.my_food_distance:
                reward -= 1
            elif new_distance < self.my_food_distance:
                reward += 1
        return reward

    # return approaching capsule reward
    def get_approaching_capsule_reward(self, gameState, action):
        reward = 0
        successor = self.getSuccessor(gameState, action)
        new_pos = successor.getAgentState(self.index).getPosition()
        if len(self.capsules_for_me) > 0:
            new_distance = min([self.getMazeDistance(new_pos, cap) for cap in self.capsules_for_me])
            if new_distance > self.my_capsule_distance:
                reward -= 1
            elif new_distance < self.my_capsule_distance:
                reward += 1
        return reward

    # return approaching drop reward
    def get_approaching_drop_reward(self, gameState, action):
        reward = 0
        successor = self.getSuccessor(gameState, action)
        new_pos = successor.getAgentState(self.index).getPosition()
        new_distance = min([self.getMazeDistance(new_pos, drop) for drop in self.drop_positions])
        if new_distance > self.current_drop_distance:
            reward -= 1
        elif new_distance < self.current_drop_distance:
            reward += 1
        return reward

    # fill self.closest_enemy_data
    def create_closest_enemy_data(self, gameState):
        min_dist = float('inf')
        for ind in self.enemy_indices:
            pos = gameState.getAgentPosition(ind)
            if pos:
                timer = gameState.getAgentState(ind).scaredTimer
                if  timer > 0 and not self.at_home(pos, 0):
                    continue
                dist = self.getMazeDistance(pos, self.my_current_position)
                if dist < min_dist:
                    min_dist = dist
                    self.closest_enemy_data = (pos, dist, timer)

    # return approaching enemy reward/penalty
    def get_approaching_enemy_reward(self, gameState, action, pos, dist):
        reward = 0
        my_new_pos = self.action_to_pos(action, self.my_current_position)
        new_dist = self.getMazeDistance(my_new_pos, pos)
        adjustment = 0
        if new_dist > dist:
            adjustment = 1
        elif new_dist < dist:
            adjustment = -1

        if gameState.getAgentState(self.index).scaredTimer == 0 and self.at_home(my_new_pos, 0):
            reward -= adjustment
        else:
            reward += adjustment
        return reward

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

    # get enemy's death flag
    def check_enemy_deaf(self, gameState, action, pos):
        my_new_pos = self.action_to_pos(action, self.my_current_position)
        if self.at_home(my_new_pos, 0) and gameState.getAgentState(self.index).scaredTimer == 0 and pos == my_new_pos:
            return True
        return False

    # check if position in our side of the board
    def at_home(self, my_pos, deep):
        if (self.red and my_pos[0] <= self.field_mid_width - deep) or (
                not self.red and my_pos[0] > self.field_mid_width + deep):
            return True
        return False

    # calculate and add reward for each turn to the reward array
    def add_reward(self):
        reward = 0
        if self.flag_done:
            reward += self.score * self.score_multiplier
        else:
            if self.flag_enemy_death:
                reward += self.enemy_death_reward
            elif self.flag_death:
                reward -= self.my_death_penalty
            else:
                if self.food_inside == 0:
                    reward += self.food_inside_prev * self.drop_food_multiplier
                if self.flag_food_eaten:
                    reward += self.food_eaten_reward

                if self.flag_enemy_around:
                    reward += self.approaching_enemy_reward * self.approaching_enemy_multiplier
                else:
                    if self.food_inside_prev >= self.stomach_size or self.prev_current_food_amount < 3:
                        reward += self.approaching_drop_reward * self.approaching_drop_multiplier
                    else:
                        reward += self.approaching_food_reward * self.approaching_food_multiplier

        self.rewards_values = np.concatenate((self.rewards_values, [reward]))

    # helper function for chooseAction
    def choose_action_probability(self, output):
        result = np.exp(output)
        result = result / np.sum(result)
        r = random.random()
        for i in range(5):
            r -= result[i]
            if r < 0:
                break
        return i

    # -=ACTION=-
    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        if self.flag_delay:
            time.sleep(0.04)

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
        if len(self.capsules_for_me) > 0:
            self.my_capsule_distance = min([self.getMazeDistance(self.my_current_position, cap) for cap in self.capsules_for_me])
        else:
            self.my_capsule_distance = float('inf')

        self.closest_enemy_data = None
        self.create_closest_enemy_data(gameState)

        state_data = np.asarray(self.state_data(gameState))
        #features = self.my_scaler.transform(state_data.reshape(1, -1))[0]
        features = state_data
        tensor_features = torch.FloatTensor(features).unsqueeze(0)
        self.online_Q_network.eval()
        result = self.online_Q_network(tensor_features).detach().numpy()[0]

        if random.random() < self.epsilon:
            self.best_action = random.choice(self.actions)
            # while True:
            #     i = self.choose_action_probability(result)
            #     self.best_action = self.index_to_action(i)
            #     if self.best_action in self.actions:
            #         break
        else:
            indices = result.argsort()[::-1]
            for ind in indices:
                self.best_action = self.index_to_action(ind.item())
                if self.best_action in self.actions:
                    break

        self.flag_death = False
        if self.my_current_position == (1, 1):
            self.flag_death = True

        self.data_set_current.append(state_data)
        self.add_reward()
        self.data_actions.append(self.best_action)

        self.flag_food_eaten = self.food_eaten_flag(gameState, self.best_action)
        if self.flag_food_eaten:
            self.food_inside += 1
        self.food_inside_prev = self.food_inside
        self.prev_current_food_amount = self.current_food_amount

        self.approaching_food_reward = self.get_approaching_food_reward(gameState, self.best_action)
        self.approaching_drop_reward = self.get_approaching_drop_reward(gameState, self.best_action)

        self.approaching_enemy_reward = 0
        self.flag_enemy_around = False
        self.flag_enemy_death = False
        if self.closest_enemy_data:
            pos, dist = self.closest_enemy_data[:2]
            self.approaching_enemy_reward = self.get_approaching_enemy_reward(gameState, self.best_action, pos, dist)
            self.flag_enemy_around = True
            self.flag_enemy_death = self.check_enemy_deaf(gameState, self.best_action, pos)


        return self.best_action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def final(self, gameState):
        self.score = self.getScore(gameState)
        self.flag_done = True

        self.data_set_current.append(self.state_data(gameState))
        all_states = np.asarray(self.data_set_current)

        #self.my_scaler.fit(all_states)
        #all_states = self.my_scaler.transform(all_states)

        actions = np.asarray(self.actions_to_indices(self.data_actions))

        self.add_reward()
        rewards = self.rewards_values[1:]

        done = np.zeros(all_states.shape[0] - 1)
        done[-1] = 1

        states = torch.FloatTensor(all_states[:-1, :])
        next_states = torch.FloatTensor(all_states[1:, :])
        actions = torch.FloatTensor(actions[1:]).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)

        states, next_states, actions, rewards, done, history = self.create_tensors_and_history(states, next_states, actions, rewards, done)

        target_Q_network = self.Duel_Q_Network()
        self.online_Q_network.train()

        # debugging
        #print('SIZE: ', done.size())
        # self.optimizer = torch.optim.Adam(self.online_Q_network.parameters(), lr=1e-4)
        #losses = []

        for epoch in range(self.epochs):
            if epoch % self.learning_step == 0:
                target_Q_network.load_state_dict(self.online_Q_network.state_dict())
            with torch.no_grad():
                online_Q_next = self.online_Q_network(next_states)
                target_Q_next = target_Q_network(next_states)
                online_max_action = torch.argmax(online_Q_next, dim=1, keepdim=True)
                y = rewards + (1 - done) * self.gamma * target_Q_next.gather(1, online_max_action.long())

            loss = F.mse_loss(self.online_Q_network(states).gather(1, actions.long()), y)

            # debugging
            #losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.total_epochs += self.epochs
        self.num_games_played += 1

        # debugging
        if self.num_games_played % 100 == 0:
            print('Total Epochs: ', self.total_epochs)
            print('Total games: ', self.num_games_played)
        #print(losses)
        #print('Reward Sum', torch.sum(rewards).item())
        #print(self.rewards_values[1:])
        #print(self.data_actions[1:])

        self.save_model(self.online_Q_network, self.optimizer, history, self.total_epochs, self.num_games_played)

    # load model
    def load_model_helper(self, side):
        file_path = 'model_' + side + '.pth'
        online_Q_network = self.Duel_Q_Network()
        optimizer = torch.optim.Adam(online_Q_network.parameters(), lr=1e-4)
        #scaler = StandardScaler()
        history = None
        epochs = 0
        games = 0
        if path.exists(file_path):
            state = torch.load(file_path)
            online_Q_network.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            #scaler = state['scaler']
            history = state['history']
            epochs = state['epochs']
            games = state['games']
        return online_Q_network, optimizer, history, epochs, games

    # save model
    def save_model_helper(self, side, model, optimizer, history, epochs, games):
        file_path = 'model_' + side + '.pth'
        my_model = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                    #'scaler': scaler,
                    'epochs': epochs,
                    'games': games}
        torch.save(my_model, file_path)

    def create_tensors_and_history(self, states, next_states, actions, rewards, done):
        if self.my_history is None:
            history = self.create_history(states, next_states, actions, rewards, done)
            return states, next_states, actions, rewards, done, history

        r_states = torch.cat((states, self.my_history['states']))
        r_next_states = torch.cat((next_states, self.my_history['next_states']))
        r_actions = torch.cat((actions, self.my_history['actions']))
        r_rewards = torch.cat((rewards, self.my_history['rewards']))
        r_done = torch.cat((done, self.my_history['done']))

        k = r_done.size(0)
        perm = torch.randperm(k)
        k = int(k / 2)
        idx = perm[:k]
        history = self.create_history(r_states[idx], r_next_states[idx], r_actions[idx], r_rewards[idx], r_done[idx])
        return r_states, r_next_states, r_actions, r_rewards, r_done, history

    def create_history(self, states, next_states, actions, rewards, done):
        return {'states': states, 'next_states': next_states, 'actions': actions, 'rewards': rewards, 'done': done}



class Agent_North(DummyAgent):
    def reward_modifiers(self):
        self.score_multiplier = 2
        self.enemy_death_reward = 10
        self.my_death_penalty = 20
        self.stomach_size = 4
        self.drop_food_multiplier = 1.5
        self.food_eaten_reward = 2
        self.approaching_enemy_multiplier = 1
        self.approaching_drop_multiplier = 1.5
        self.approaching_food_multiplier = 1.5

    def get_my_food_positions(self):
        n = int(self.current_food_amount / 2)
        return self.current_food_positions[n:]

    def load_model(self):
        side = 'North'
        return self.load_model_helper(side)

    def save_model(self, model, optimizer, history, epochs, games):
        side = 'North'
        self.save_model_helper(side, model, optimizer, history, epochs, games)


class Agent_South(DummyAgent):
    def reward_modifiers(self):
        self.score_multiplier = 1
        self.enemy_death_reward = 20
        self.my_death_penalty = 10
        self.stomach_size = 3
        self.drop_food_multiplier = 2
        self.food_eaten_reward = 1
        self.approaching_enemy_multiplier = 2
        self.approaching_drop_multiplier = 1
        self.approaching_food_multiplier = 0.5

    def get_my_food_positions(self):
        n = int((self.current_food_amount + 1) / 2)
        return self.current_food_positions[:n]

    def load_model(self):
        side = 'South'
        return self.load_model_helper(side)

    def save_model(self, model, optimizer, history, epochs, games):
        side = 'South'
        self.save_model_helper(side, model, optimizer, history, epochs, games)


class Duel_Q_Network_1000(nn.Module):
    def __init__(self):
        super(Duel_Q_Network_1000, self).__init__()

        self.fc1 = nn.Linear(1125, 800)
        self.fc2 = nn.Linear(800, 512)

        self.fc_value = nn.Linear(512, 128)
        self.fc_adv = nn.Linear(512, 128)

        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 5)

        self.a_func = nn.Sigmoid()
        #self.a_func = nn.LeakyReLU()

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight)

    def forward(self, state):
        y = self.a_func(self.fc1(state))
        y = self.a_func(self.fc2(y))

        value = self.a_func(self.fc_value(y))
        adv = self.a_func(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_average

        return Q

class Duel_Q_Network_simple(nn.Module):
    def __init__(self):
        super(Duel_Q_Network_simple, self).__init__()

        self.fc1 = nn.Linear(31, 42)
        self.fc2 = nn.Linear(42, 21)

        self.fc_value = nn.Linear(21, 7)
        self.fc_adv = nn.Linear(21, 11)

        self.value = nn.Linear(7, 1)
        self.adv = nn.Linear(11, 5)

        self.a_func = nn.Tanh()
        #self.a_func = nn.Sigmoid()
        #self.a_func = nn.LeakyReLU()

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight)

    def forward(self, state):
        y = self.a_func(self.fc1(state))
        y = self.a_func(self.fc2(y))

        value = self.a_func(self.fc_value(y))
        adv = self.a_func(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_average

        return Q

class Duel_Q_Network_very_simple(nn.Module):
    def __init__(self):
        super(Duel_Q_Network_very_simple, self).__init__()

        #self.fc1 = nn.Linear(31, 21)

        self.fc_value = nn.Linear(27, 9)
        self.fc_adv = nn.Linear(27, 18)

        self.value = nn.Linear(9, 1)
        self.adv = nn.Linear(18, 5)

        self.a_func = nn.Tanh()
        # self.a_func = nn.Sigmoid()
        # self.a_func = nn.LeakyReLU()

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight)

    def forward(self, state):
        #y = self.a_func(self.fc1(state))
        value = self.a_func(self.fc_value(state))
        adv = self.a_func(self.fc_adv(state))
        value = self.value(value)
        adv = self.adv(adv)

        adv_average = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - adv_average

        return Q
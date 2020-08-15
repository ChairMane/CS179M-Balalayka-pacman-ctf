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
import pandas as pd
import numpy as np
import functools
import operator


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
        self.prev_enemy_food_amount = self.get_enemy_food_amount(gameState)
        self.flag_eat_mode = True
        self.drop_positions = self.get_drop_positions(gameState)
        self.current_food_positions = []
        self.flag_food_eaten = False
        self.flag_food_eaten_prev = False
        self.flag_death = False

        self.data_grid_radius = 5
        self.features_groups = 9
        self.qualities = 13
        self.data_set_current = []

        self.epsilon = 0
        self.gamma = 0.99
        self.penalty = 0.1

        self.states_values = np.empty(0)
        self.rewards_values = np.empty(0)
        self.prev_state_value = 0
        self.flag_win = False
        self.flag_lose = False

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

    def create_state_data_v1(self, gameState):
        # food, drop predicted
        food_future_dist = np.full(5, 10 / (self.my_food_distance + 1))
        drop_future_dist = np.full(5, 10 / (self.current_drop_distance + 1))
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
                grid_qualities[9 + i] = dist
                (x_t, y_t) = pos
                if x_0 <= x_t <= x_1 and y_0 <= y_t <= y_1:
                    grid_positions[7 + i, (y_t - y_0) * n + x_t - x_0] = 1
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

        return np.concatenate((food_future_dist, drop_future_dist, enemy_future_dist.ravel(), grid_positions.ravel(), grid_qualities))

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

    def get_enemy_food_amount(self, gameState):
        blue_food = gameState.getBlueFood().asList()
        red_food = gameState.getRedFood().asList()
        if self.red:
            return len(blue_food)
        else:
            return len(red_food)

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

    def state_value(self, gameState):
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

    def update_reward(self):
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
                        self.data_value[-6:] += reward * 1.5
                    else:
                        self.data_value += reward[-n:]
                if self.flag_food_eaten_prev:
                    if n >= 6:
                        self.data_value[-6:] += reward * 1.5
                    else:
                        self.data_value += reward[-n:]
            #if n > 1:
                #self.data_value[-2] += 0.6 * (self.data_value[-1] - self.data_value[-2])
            # How to check enemy's death????

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
            reward += self.enemy_food_amount - self.prev_enemy_food_amount
        self.rewards_values = np.concatenate((self.rewards_values, [reward]))

    def calc_returns(self, rewards):
        n = rewards.shape[0]
        returns = np.zeros(n)
        for i in range(n):
            for j in range(n - i):
                returns[i] += rewards[i + j] * self.gamma**j
        return returns

    def state_value_differences(self):
        n = self.states_values.shape[0] - 1
        result = np.empty(n)
        for i in range(n):
            result[i] = self.states_values[i + 1] - self.states_values[i]
        return result


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        #time.sleep(0.06)

        self.actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''

        self.food_inside_prev = self.food_inside
        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.is_home = self.at_home(self.my_current_position, 0)
        if self.is_home:
            self.food_inside = 0

        self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])

        self.current_food_positions, self.enemy_food_positions, self.capsules_for_me, self.capsules_for_enemy = self.all_food_positions(gameState)
        self.enemy_food_amount = len(self.enemy_food_positions)

        self.current_food_positions.sort(key = lambda x: x[1])
        self.current_food_amount = len(self.current_food_positions)
        self.my_food_positions = self.get_my_food_positions()
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])
        else:
            self.my_food_distance = float('inf')


        self.current_enemy_distances_positions = []
        self.closest_enemy_distance = float('inf')
        for index in self.enemy_indices:
            pos = gameState.getAgentPosition(index)
            if pos:
                if gameState.getAgentState(index).scaredTimer > 3 and not self.at_home(pos, 0):
                    continue
                dist = self.getMazeDistance(self.my_current_position, pos)
                if dist < self.closest_enemy_distance:
                    self.closest_enemy_distance = dist
                self.current_enemy_distances_positions.append((dist, pos))

        pacman_stomach_size = 5
        if self.my_food_distance < 3:
            pacman_stomach_size += 1

        if self.current_food_amount < 3 or self.food_inside > pacman_stomach_size or (self.food_inside > 1 and self.current_drop_distance < 6):
            self.flag_eat_mode = False
        else:
            self.flag_eat_mode = True

        action_value = float('-inf')
        best_action = 'Stop'
        if random.random() < self.epsilon:
            #while best_action == 'Stop':
            best_action = random.choice(self.actions)
        else:
            for action in self.actions:
                if action == 'Stop':
                    new_action_value = -0.3
                else:
                    new_action_value = self.action_value(gameState, action)

                if new_action_value > action_value:
                    best_action = action
                    action_value = new_action_value

        self.flag_food_eaten = self.food_eaten_flag(gameState, best_action)
        if self.flag_food_eaten:
            self.food_inside += 1

        self.flag_death = False
        if self.my_current_position == (1, 1):
            self.flag_death = True


        state_data = self.create_state_data_simple(gameState)

        self.data_set_current.append(np.concatenate((state_data, self.add_move(best_action))))
        #self.states_values = np.concatenate((self.states_values, [self.state_value(gameState)]))
        self.add_reward()

        self.flag_food_eaten_prev = self.flag_food_eaten
        self.prev_enemy_food_amount = self.enemy_food_amount

        return best_action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def recede_home(self, next_pos):
        new_dist = min([self.getMazeDistance(next_pos, drop) for drop in self.drop_positions])
        if self.current_drop_distance < new_dist:
            return True
        return False

    def at_home(self, my_pos, deep):
        if (self.red and my_pos[0] <= self.field_mid_width - deep) or (
                not self.red and my_pos[0] > self.field_mid_width + deep):
            return True
        return False

    def get_my_food_positions(self):
        return self.current_food_positions

    def enemy_distance_value(self, dist_current, dist_next):
        return math.log2((dist_next + 1) / (dist_current + 1))

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

    def food_hunting(self, my_next_pos):
        value_h = 0
        shift = 0.1
        next_f_dist = min([self.getMazeDistance(my_next_pos, food) for food in self.my_food_positions])
        if next_f_dist < self.my_food_distance:
            value_h += shift
        elif next_f_dist > self.my_food_distance:
            value_h -= shift
        return value_h

    def food_depositing(self, my_next_pos):
        value_d = 0
        shift = 0.1
        next_d_dist = min([self.getMazeDistance(my_next_pos, drop) for drop in self.drop_positions])
        if next_d_dist < self.current_drop_distance:
            value_d += shift
        elif next_d_dist > self.current_drop_distance:
            value_d -= shift
        return value_d

    def patrol(self, my_next_pos):
        value_p = random.random() / 10
        if not self.at_home(my_next_pos, 0):
            value_p = -0.1
        return value_p

    def action_value(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        value = 0

        my_next_position = successor.getAgentState(self.index).getPosition()
        next_is_home = self.at_home(my_next_position, 0)
        if self.flag_eat_mode:
            hunting_value = self.food_hunting(my_next_position)
            value += hunting_value
        elif self.food_inside > 0:
            value += self.food_depositing(my_next_position)
        else:
            value += self.food_depositing(my_next_position) / 10 + self.patrol(my_next_position)

        enemy_positions_value = 0
        for item in self.current_enemy_distances_positions:
            current_enemy_distance, current_enemy_position = item
            next_enemy_distance = self.getMazeDistance(my_next_position, current_enemy_position)
            enemy_position_change = self.enemy_distance_value(current_enemy_distance, next_enemy_distance) / current_enemy_distance

            enemy_is_home = self.at_home(current_enemy_position, 0)
            if gameState.getAgentState(self.index).scaredTimer > 0:
                enemy_positions_value += enemy_position_change
            else:
                if enemy_is_home:
                    if self.is_home:
                        enemy_positions_value -= enemy_position_change * 4
                    else:
                        if next_is_home:
                            enemy_positions_value -= enemy_position_change
                        else:
                            if next_enemy_distance == 1:
                                enemy_positions_value += enemy_position_change
                            else:
                                enemy_positions_value -= enemy_position_change
                else:
                    enemy_positions_value += enemy_position_change * 2

        value += enemy_positions_value

        if not self.is_home and self.recede_home(my_next_position):
            value -= 0.06
            if self.closest_enemy_distance < 6:
                value -= 0.4 / self.closest_enemy_distance
                if len(successor.getLegalActions(self.index)) == 2:
                    value -= 0.2

        return value


class Agent_North(DummyAgent):
    def get_my_food_positions(self):
        n = int(self.current_food_amount / 2)
        return self.current_food_positions[n:]

    def final(self, gameState):
        if gameState.data.score > 0:
            self.flag_win = True
        if gameState.data.score < 0:
            self.flag_lose = True
        self.add_reward()
        rev = self.rewards_values[1:]
        #print(rev)
        #print(self.calc_returns(rev))
        # df = pd.DataFrame(np.column_stack((np.asarray(self.data_set_current), self.calc_returns(rev))))
        # df.to_csv('my_data_North.csv', mode='a', header=False, index=False)


class Agent_South(DummyAgent):
    def get_my_food_positions(self):
        n = int((self.current_food_amount + 1) / 2)
        return self.current_food_positions[:n]

    def final(self, gameState):
        if gameState.data.score > 0:
            self.flag_win = True
        if gameState.data.score < 0:
            self.flag_lose = True
        self.add_reward()
        rev = self.rewards_values[1:]
        # df = pd.DataFrame(np.column_stack((np.asarray(self.data_set_current), self.calc_returns(rev))))
        # df.to_csv('my_data_South.csv', mode='a', header=False, index=False)
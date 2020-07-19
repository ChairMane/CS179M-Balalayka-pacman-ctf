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
        self.flag_eat_mode = True
        self.drop_positions = self.get_drop_positions(gameState)
        self.current_food_positions = []
        self.score = 0

        self.data_grid_radius = 5
        self.features_groups = 9
        self.qualities = 9
        self.data_set_current = []

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

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        #time.sleep(0.06)

        # machine learning data collection

        self.collecting_data(gameState)

        # end data collection

        global best_action
        actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''

        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.is_home = self.at_home(self.my_current_position, 0)
        if self.is_home:
            self.food_inside = 0
        else:
            self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])

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
        self.my_food_positions = self.get_my_food_positions()
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])


        self.current_enemy_distances_positions = []
        self.closest_enemy_distance = 1000
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

        action_value = -10
        flag_food_eaten = False
        for action in actions:
            if action == 'Stop':
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

        state_data = self.create_state_data(gameState)

        self.data_set_current.append(np.concatenate((state_data, self.add_move(best_action))))

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

    def object_distance_value(self, dist_current, dist_next):
        return 0.003 * (dist_next**3 - dist_current**3)

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
        flag_h = False
        next_f_dist = min([self.getMazeDistance(my_next_pos, food) for food in self.my_food_positions])
        if next_f_dist < self.my_food_distance:
            value_h += shift
            if next_f_dist == 0:
                flag_h = True
        elif next_f_dist > self.my_food_distance:
            value_h -= shift
        return value_h, flag_h

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

        flag_food_taken = False

        my_next_position = successor.getAgentState(self.index).getPosition()
        next_is_home = self.at_home(my_next_position, 0)
        if self.flag_eat_mode:
            hunting_value, flag_food_taken = self.food_hunting(my_next_position)
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

        return value, flag_food_taken

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
                df.to_csv('my_data_North.csv', mode='a', header=False, index=False)
            del self.data_set_current[:]

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
                df.to_csv('my_data_South.csv', mode='a', header=False, index=False)
            del self.data_set_current[:]

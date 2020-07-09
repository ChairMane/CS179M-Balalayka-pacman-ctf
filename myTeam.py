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

def createTeam(firstIndex, secondIndex, isRed, first='DummyAgent', second='UpFucker'):
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

        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        self.my_current_position = gameState.getAgentState(self.index).getPosition()
        self.current_food_positions = self.getFood(gameState).asList()
        self.current_food_amount = len(self.current_food_positions)
        self.my_food_positions = self.get_my_food_positions(gameState)
        if len(self.my_food_positions) > 0:
            self.my_food_distance = min([self.getMazeDistance(self.my_current_position, food) for food in self.my_food_positions])
        self.drop_positions = self.get_drop_positions(gameState)
        self.current_drop_distance = min([self.getMazeDistance(self.my_current_position, drop) for drop in self.drop_positions])

        self.enemy_indices = self.getOpponents(gameState)
        self.current_enemy_positions = []
        self.current_enemy_distanses_positions = []
        for index in self.enemy_indices:
            pos = gameState.getAgentPosition(index)
            if pos:
                self.current_enemy_distanses_positions.append((self.getMazeDistance(self.my_current_position, pos), pos))


        pacman_stomach_size = int(self.current_food_amount / 4 + 1)

        if self.my_current_position[0] < 16:
            self.food_inside = 0
        if self.current_food_amount < 3 or self.food_inside > pacman_stomach_size:
            self.flag_eat_mode = False
        else:
            self.flag_eat_mode = True



        action_value = -10
        flag_food_eaten = False
        for action in actions:
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

    def action_value(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        value = 0
        if action is 'Stop':
            value -= 0.1
        flag_food_taken = False

        my_next_position = successor.getAgentState(self.index).getPosition()

        if self.flag_eat_mode:
            next_food_distance = min([self.getMazeDistance(my_next_position, food) for food in self.my_food_positions])
            if next_food_distance is 0:
                next_food_distance = 0.5
                flag_food_taken = True
            value += 1 / next_food_distance - 1 / self.my_food_distance
        elif self.food_inside > 0:
            next_drop_distance = min([self.getMazeDistance(my_next_position, drop) for drop in self.drop_positions])
            if next_drop_distance is 0:
                self.food_inside = 0
                self.flag_eat_mode = True
                return 5, False
            value += 1 / next_drop_distance
            value -= 1 / self.current_drop_distance

        if my_next_position is (1, 1):
            value -= 2

        enemy_positions_value = 0
        closest_enemy_distance = 1000
        for item in self.current_enemy_distanses_positions:
            current_enemy_distance = item[0]
            current_enemy_position = item[1]
            next_enemy_distance = self.getMazeDistance(my_next_position, current_enemy_position)
            if next_enemy_distance is 0:
                next_enemy_distance = 0.5
                if current_enemy_position[0] < 16:
                    value += 1
            if current_enemy_distance < closest_enemy_distance:
                closest_enemy_distance = current_enemy_distance
            enemy_positions_value += 1 / current_enemy_distance - 1 / next_enemy_distance

        if my_next_position[0] > 14:
            value += enemy_positions_value
            if len(successor.getLegalActions(self.index)) is 2:
                value -= 0.9
                if closest_enemy_distance < 4:
                    value -= 1
        else:
            value -= enemy_positions_value * 1.3

        return value, flag_food_taken


class UpFucker(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, action) for action in actions]
        max_value = max(values)

        # TODO:
        # Check out better evaluation function
        avg_value = int(sum(values) / len(values))
        decent_actions = [(action, value) for action, value in zip(actions, values) if value in self.getDecentRange(max_value, avg_value)]
        decent_actions.sort()
        best_action = min(decent_actions)[0]
        return best_action

    def getDecentRange(self, value, avg_value):
        return range((value - avg_value), (value + avg_value))

    def evaluate(self, gameState, action):

        features = self.getFeatures(gameState, action)
        weights = self.getWeights()
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        weights = self.getWeights()
        successor = gameState.generateSuccessor(self.index, action)
        my_position = successor.getAgentState(self.index).getPosition()

        #_______ Food features __________
        food_distances = self.getFoodDistances(successor, my_position)
        features['foodDistance'] = food_distances[0]

        #_______ Enemy features _________
        enemy_distances = self.getEnemyFuckers(successor,  my_position)
        features['enemyDistance'] = enemy_distances


        return features

    def getFoodDistances(self, successor, my_position):
        food_list = self.getFood(successor).asList()
        food_distances = [self.getMazeDistance(my_position, food) for food in food_list]
        food_distances.sort()
        return food_distances

    def getEnemyFuckers(self, successor, my_position):
        opponents = self.getOpponents(successor)
        distances = []
        noisy_distances = [(dis, index) for index, dis in enumerate(successor.getAgentDistances()) if index in opponents]
        for opp in opponents:
            enemy_position = successor.getAgentPosition(opp)
            if enemy_position:
                distances.append((self.getMazeDistance(my_position, enemy_position), opp))
            else:
                distances.append((9999, opp))

        return min(noisy_distances)[0] if all(v == 9999 for v, i in distances) else min(distances)[0]

    def getWeights(self):
        return {'foodDistance' : 100, 'enemyDistance' : -1}

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


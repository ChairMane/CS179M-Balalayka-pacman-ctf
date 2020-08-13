# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import os.path
import pickle
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

NUM_GAMES = 0

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveReflexAgent', second='OffensiveReflexAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]



####################
# Torch Essentials #
####################

#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Transition():

    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

# This class will hold the history of states, actions, rewards and next states
# This will be used to train the agent
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

global_memory0 = ReplayMemory(50000)
global_memory2 = ReplayMemory(50000)

class DQN(nn.Module):

    def __init__(self, feature_size, action_size):
        super(DQN, self).__init__()
        self.input = nn.Linear(feature_size, 12)
        self.hidden1 = nn.Linear(12, 5)
        self.hidden2 = nn.Linear(5, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.softmax(self.hidden2(x), dim=1)
        return x

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.policy_net = self.loadModel(gameState, 'models/policy{}.pt'.format(self.index))
        self.target_net = self.loadModel(gameState, 'models/target{}.pt'.format(self.index))
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.capacity = 50000
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = self.loadReplayMemory('memory{}.txt'.format(self.index))
        self.steps_done = 0
        self.action_space = self.mapActions()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        global NUM_GAMES
        if self.index == 0:
            NUM_GAMES += 1
            print('Number of games so far', NUM_GAMES)

    def final(self, gameState):
        if self.index == 0:
            global global_memory0
            if (global_memory0.position + 300) > self.capacity:
                print('Starting to write index 0 to file...')
                self.saveReplayMemory(global_memory0.memory, 'memory{}.txt'.format(self.index))
        elif self.index == 2:
            global global_memory2
            if (global_memory2.position + 300) > self.capacity:
                print('Starting to write index 2 to file...')
                self.saveReplayMemory(global_memory2.memory, 'memory{}.txt'.format(self.index))

    def createModel(self, feature_size, action_size):
        return DQN(feature_size, action_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = [*zip(*transitions)]

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch[2])), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch[2]
                                           if s is not None])
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[3])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        self.current_food_position = self.all_food_positions(gameState)
        self.current_food_amount = len(self.current_food_position)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        bestAction = ''
        network_action = None
        next_state = []
        if sample > eps_threshold:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            with torch.no_grad():
                input = torch.FloatTensor(list(self.getFeatures(gameState, bestAction).values())).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                policy_net = self.policy_net(input)
                next_best = policy_net.max(1)[1].view(1, 1).item()

                if self.action_space[next_best] not in actions:
                    i = 1
                    while self.action_space[next_best] not in actions:
                        next_best = policy_net.topk(5)[1][0][i].item()
                        i += 1
                    bestAction = self.action_space[next_best]
                else:
                    bestAction = self.action_space[next_best]

            key_list = list(self.action_space.keys())
            val_list = list(self.action_space.values())
            best_action_index = key_list[val_list.index(bestAction)]
            network_action = torch.tensor([[best_action_index]], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.long)
        else:
            bestAction = random.choice(bestActions)
            key_list = list(self.action_space.keys())
            val_list = list(self.action_space.values())
            best_action_index = key_list[val_list.index(bestAction)]
            network_action = torch.tensor([[best_action_index]], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.long)

        state = torch.FloatTensor(list(self.getFeatures(gameState, bestAction).values())).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        next_state = torch.FloatTensor(self.getNextState(self.getSuccessor(gameState, bestAction))).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        reward = self.getRewards(gameState, bestAction)
        reward = torch.tensor([reward], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.memory.push(state, network_action, next_state, reward)

        if self.index == 0:
            global global_memory0
            global_memory0.push(state, network_action, next_state, reward)
        elif self.index == 2:
            global global_memory2
            global_memory2.push(state, network_action, next_state, reward)

        self.optimize_model()
        if NUM_GAMES % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.saveModel(self.target_net, 'target{}.pt'.format(self.index))
            self.saveModel(self.policy_net, 'policy{}.pt'.format(self.index))
        return bestAction

    def saveReplayMemory(self, memory, filename):
        with open('models/{}'.format(filename), 'wb') as fp:
            pickle.dump(memory, fp)
        fp.close()

    def loadReplayMemory(self, filename):
        if os.path.isfile('models/{}'.format(filename)):
            with open('models/{}'.format(filename), 'rb') as fp:
                memory_list = pickle.load(fp)
            fp.close()
            memory = ReplayMemory(50000)
            memory.memory = memory_list
            memory.position = len(memory_list)
            return memory
        else:
            if self.index == 0:
                global global_memory0
                if len(global_memory0) > 0:
                    return global_memory0
                return ReplayMemory(50000)
            elif self.index == 2:
                global global_memory2
                if len(global_memory2) > 0:
                    return global_memory2
                return ReplayMemory(50000)

    def saveModel(self, network, filename):
        torch.save(network.state_dict(), 'models/{}'.format(filename))

    def loadModel(self, gameState, filename):
        model = self.createModel(len(self.getWeights(gameState, 'testing')), 5)

        if os.path.isfile(filename):
            model.load_state_dict(torch.load(filename))
            model.eval()
            # print('Model loaded...')
            return model
        else:
            # print('No model loaded... Creating new model...')
            return model

    def getNextState(self, successor):
        actions = successor.getLegalActions(self.index)
        values = [self.evaluate(successor, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        return list(self.getFeatures(successor, bestAction).values())


    def mapActions(self):
        return {0 : 'Stop', 1 : 'North', 2 : 'South', 3 : 'West', 4: 'East'}

    def getRewards(self, gameState, bestAction):
        food_consumed = self.food_eaten_flag(gameState, bestAction)
        if food_consumed:
            return 1
        else:
            return 0

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

    def all_food_positions(self, gameState):
        blue_food = gameState.getBlueFood().asList()
        red_food = gameState.getRedFood().asList()
        if self.red:
            current_food_positions = blue_food
        else:
            current_food_positions = red_food
        return current_food_positions

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        else:
            features['invaderDistance'] = -1000

        if action == Directions.STOP: features['stop'] = 1
        else: features['stop'] = 0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        else: features['reverse'] = 0

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

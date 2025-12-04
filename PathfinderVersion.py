# oversleepersTeam.py
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
from util import nearestPoint
import pdb
import uuid
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
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
        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        self.center_line = self.compute_center_line(gameState)
        CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)

    def compute_center_line(self, gameState):
        """
        Return center line positions.
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2 - 1
        else:
            border_x = self.arena_width // 2
        border_line = [(border_x, h) for h in range(self.arena_height)]
        # return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls]


EXPLORE_RATE = 2.0
MAX_INTERATIONS = 10
MAX_SIMULATE_STEPS = 0
MAX_DEPTH = 1

class MCTS:
    def __init__(self, gameState, agent, evaluateFun, action, current_node = None):
        self.root = MCTSNode(gameState, agent, action, current_node)
        # self.nodes = {}
        # self.nodes[hash(gameState)] = self.root 
        self.evaluateFun = evaluateFun

        self.gameState = gameState.deepCopy()
        self.agent = agent
        self.index = agent.index
        self.search_depth = 0
        
    def select(self, node):
        if self.is_fully_expanded(node) == False:
            self.fully_expand(node)
        best_child =self.select_best_child(node)
        
        return best_child
    
    def is_fully_expanded(self, node):
        if len(node.children)==len(node.legalActions):
            return True
        return False
    
    def fully_expand(self, node):
        children_actions = [child.action for child in node.children]
        legal_actions = node.legalActions
        for action in [x for x in legal_actions if x not in children_actions]:
            current_game_state = node.gameState.deepCopy()
            next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
            child = MCTSNode(next_game_state, self.agent, action, node)
            node.addChild(child) 
        return node
    
    def select_best_child(self, node):
        if self.search_depth >= MAX_DEPTH:
            return node
        
        total_visits = self.root.visit
        best_child = node
        best_score = -np.inf

        for child in node.children:
            if child.visit == 0:
                best_child = child
                return best_child

            score = child.compute_UCB(total_visits)
            if score > best_score:
                best_score = score
                best_child = child
            
            self.expand(best_child)
            
        self.search_depth += 1
        return self.select_best_child(best_child)

    def expand(self, node):
        """
        Expand the given node by one of its unexplored actions.
        Return the newly expanded child node.
        """
        if node.unexploredActions != []:
            current_game_state = node.gameState.deepCopy()
            legal_actions = current_game_state.getLegalActions(self.agent.index)
            action = node.unexploredActions.pop()
            while action not in legal_actions:
                if node.unexploredActions == []:
                    break
                action = node.unexploredActions.pop()
            if action in legal_actions:
                next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
                child = MCTSNode(next_game_state, self.agent, action, node)
                node.addChild(child) 
                # TODO can not reuse node only based on gameState
                # self.nodes[hash(next_game_state)] = child
                return child
            else:
                return None
        elif len(node.children) == 0:
            node.update_legal_actions()
            self.expand(node)
        else:
            return node

    def simulate(self, node):
        simulated_game_state = node.gameState.deepCopy()
        current_game_state = node.gameState.deepCopy()
        steps = 0
        action = None
        while steps < MAX_SIMULATE_STEPS:
            available_actions = simulated_game_state.getLegalActions(self.index)
            if available_actions: 
                random_action = random.choice(available_actions)
                current_game_state = simulated_game_state
                simulated_game_state = simulated_game_state.generateSuccessor(self.index, random_action)
                action = random_action
            steps += 1
        
        if action:
            return self.evaluateFun(current_game_state.deepCopy(), random_action)
        else: 
            return self.evaluateFun(node.parent.gameState.deepCopy(), node.action)
    
    def cal_reward(self, gameState):
        current_pos = gameState.getAgentPosition(self.agent.index)
        if current_pos == gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_feature(gameState) * self.get_weight_backup()
        return value
    
            
    def backpropagation(self, node, reward):
        node.visit += 1
        node.value += reward
        if node.parent is not None:
            self.backpropagation(node.parent, reward)

    def perform(self):
        # must move when reach the time limit
        timeLimit = 0.99
        
        start = time.time()
        node = self.root
        count = 0
        while (time.time() - start < timeLimit) and count < MAX_INTERATIONS:
            child = self.select(node)
            if child is None:
                pass
            else:
                reward = self.simulate(child)
                self.backpropagation(child, reward)
            count += 1
            self.search_depth = 0

        # TODO choose the method of choose action, max visit or max value
        if len(self.root.children) > 0:
            return self.select_action_by_visit()
        else:
            legal_actions = self.gameState.getLegalActions(self.agent.index)
            return random.choice(legal_actions)
    
    def select_action_by_visit(self):
        best_child = max(self.root.children, key=lambda child: child.visit)
        best_action = best_child.action
        return best_action
    
    def select_action_by_value(self):
        best_child = max(self.root.children, key=lambda child: child.value / child.visit)
        best_action = best_child.action

        return best_action
            
    def update_root(self, new_state):
        """
        upate the root of the tree
        """
        new_root = MCTSNode(new_state, self.agent , None, self.root)
        self.root = new_root
        self.gameState = new_state.deepCopy()
        
##########
#  MCTS  #
##########

class MCTSNode (): 
    def __init__(self, gameState, agent, action, parent=None):
        self.gameState = gameState
        self.parent = parent 
        self.children = []
        self.visit = 0
        self.value = 0
        self.action = action
        self.agent = agent
        self.depth = self.depth = parent.depth + 1 if parent else 0
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        
    def update_legal_actions(self):
        self.legalActions = [act for act in self.gameState.getLegalActions(self.agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        
    def compute_UCB(self, N):
        """
        N: Total visits
        """
        exploitation = self.value/self.visit
        explore = EXPLORE_RATE * math.sqrt(math.log(N)/self.visit)  
        score= exploitation + explore
        
        return score
    
    def addChild(self, child):
        self.children.append(child)

####################
#  OffensiveAgent  #
####################
class OffensiveAgent(DummyAgent): 
    def registerInitialState(self, gameState):
        self.MCTS_tree = None
        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        DummyAgent.registerInitialState(self, gameState)        
        self.friendly_borders = self.detect_my_border(gameState)
        self.hostile_borders = self.detect_enemy_border(gameState)
        
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
    
    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        carrying = agent_state.numCarrying
        isPacman = agent_state.isPacman
        
        if isPacman:
            if self.MCTS_tree is None:
                self.MCTS_tree = MCTS(gameState, self, self.evaluate_off, None,  None)
            else:
                self.MCTS_tree.update_root(gameState)
        else:
            if self.MCTS_tree is None:
                self.MCTS_tree = MCTS(gameState, self, self.evaluate_def, None,  None)
            else:
                self.MCTS_tree.update_root(gameState)
        return self.MCTS_tree.perform()
    
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onOffense'] = 0
        if myState.isPacman: features['onOffense'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(defenders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
            minDistance = min(dists)
            # keep a distance from defenders
            if(minDistance < 10) and all(enemy.scaredTimer == 0 for enemy in defenders):    
                features['invaderDistance'] = minDistance
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        
        current_pos = gameState.getAgentPosition(self.index)
        features['distance'] = min([self.getMazeDistance(current_pos, borderPos) for borderPos in self.center_line])
        
        foodList = self.getFood(successor).asList() 
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        if len(foodList) == 0:
            myPos = successor.getAgentState(self.index).getPosition()
            distance_to_home = self.getMazeDistance(myPos, gameState.getInitialAgentPosition(self.index))
            features['distanceToHome'] = distance_to_home
        # compute the num carrying
        features['numCarrying'] = successor.getAgentState(self.index).numCarrying
            
        # if carrying too much score, then get close to home
        if 1 < successor.getAgentState(self.index).numCarrying < 7:
        # compute the distance to the start point
            features['distanceToHomeCarryingTooMuch'] = self.getMazeDistance(current_pos, gameState.getInitialAgentPosition(self.index))
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)#self.getScore(successor)
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['enemyScaredTime'] = sum([enemy.scaredTimer for enemy in invaders])
        # get closed to scared enemy
        if len(invaders) > 0 and any(enemy.scaredTimer > 0 for enemy in invaders):
            # compute the distance to the scared enemy
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            minDistance = min(dists)
            features['scaredEnemyDistance'] = minDistance

        # compute the distance to the nearest capsule
        capsules = self.getCapsules(successor)
        if capsules:
            capsule_distances = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
            nearest_capsule_distance = min(capsule_distances)
            features['nearestCapsuleDistance'] = nearest_capsule_distance
        
        return features 

    def getWeights(self, gameState, action):
        return {'onOffense': 1000, 'successorScore': 1000, 'distanceToFood': -50, 'distance': -10, 'stop': -100, 'reverse': -2, 'invaderDistance': 100, 'distanceToHome': -100, 'enemyScaredTime': 100, 'scaredEnemyDistance': -100, 'distanceToHomeCarryingTooMuch': -100, 'numCarrying': 10, 'nearestCapsuleDistance': -100}
    
    def evaluate (self, gameState, action):
        current_pos = gameState.getAgentPosition(self.index)
        if current_pos == gameState.getInitialAgentPosition(self.index):
            return -1000
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights
    
    def detect_my_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2 - 1
        else:
            border_x = self.arena_width // 2
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2
        else:
            border_x = self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_ghost(self, gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList

    def detect_enemy_approaching(self, gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        dangerGhosts = []
        ghosts = self.detect_enemy_ghost(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for g in ghosts:
            distance = self.getMazeDistance(myPos, gameState.getAgentPosition(g))
            if distance <= 5:
                dangerGhosts.append(g)
        return dangerGhosts

    def evaluate_off(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_off_features(gameState, action)
        weights = self.get_off_weights(gameState, action)
        return features * weights

    def get_off_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_tate = self.get_next_state(gameState, action)
        if next_tate.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_tate).asList()) > 0:
                features['minDistToFood'] = self.get_min_dist_to_food(next_tate)
        return features

    def get_off_weights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1, 'getFood': 100}

    def evaluate_def(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_def_features(gameState, action)
        weights = self.get_def_weights(gameState, action)
        return features * weights

    def get_def_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_next_state(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_def_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def get_min_dist_to_food(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(myPos, f) for f in self.getFood(gameState).asList()])
####################
#  DefensiveAgent  #
####################

class DefensiveAgent(DummyAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    
    def registerInitialState(self, gameState):
        self.MCTS_tree = None
        
        DummyAgent.registerInitialState(self, gameState)
    
    def chooseAction(self, gameState):
        if self.MCTS_tree is None:
            self.MCTS_tree = MCTS(gameState, self, self.evaluate, None,  None)
        else:
            self.MCTS_tree.update_root(gameState)
        return self.MCTS_tree.perform()
    
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
        
    def evaluate (self, gameState, action):
        current_pos = gameState.getAgentPosition(self.index)
        if current_pos == gameState.getInitialAgentPosition(self.index):
            return -1000
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

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
        if len(invaders) > 0 and all(enemy.scaredTimer == 0 for enemy in invaders):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        # sum up enemy scared time
        
        features['enemyScaredTime'] = sum([enemy.scaredTimer for enemy in invaders])
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        
        current_pos = gameState.getAgentPosition(self.index)
        features['distance'] = min([self.getMazeDistance(current_pos, borderPos) for borderPos in self.center_line])
        
        foodList = self.getFoodYouAreDefending(successor).asList()    
        features['foodDefendingScore'] = len(foodList)
        
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        # get closed to scared enemy
        if len(invaders) > 0 and any(enemy.scaredTimer > 0 for enemy in invaders):
            # compute the distance to the scared enemy
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            minDistance = min(dists)
            features['scaredEnemyDistance'] = minDistance
        
        return features 

    def getWeights(self, gameState, action):
        # TODO adjust reward policy
        return {'numInvaders': -1000, 'onDefense': 1000, 'invaderDistance': -50, 'stop': -100, 'reverse': -2, 'distance': -2, 'foodDefendingScore': 10, 'distanceToFood': -1, 'enemyScaredTime': 1000, 'scaredEnemyDistance': -100}
  
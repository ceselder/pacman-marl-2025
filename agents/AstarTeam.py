# Problem -> Solver -> Plan
#
# Problem = <S, s_0, S_g, A(s), T(a,s) -> s', C(a,s) -> int>
# Solver = explores the problem state space, finding a plan from the initial state to goal
# Plan = The list of actions to perform

import threading
import util
from contextlib import contextmanager
import signal
import typing as t

# Standard imports
from captureAgents import CaptureAgent
import distanceCalculator
import random
from game import Directions, Actions  # basically a class to store data
import game

# My imports
from capture import GameState
import logging

# this is the entry points to instanciate you agents
def createTeam(firstIndex: int, secondIndex: int, isRed: bool,
               first: str = 'offensiveAgent', second: str = 'defensiveAgent') -> t.List[CaptureAgent]:

    # capture agents must be instanciated with an index
    # time to compute id 1 second in real game
    return [eval(first)(firstIndex, timeForComputing=1), eval(second)(secondIndex, timeForComputing=1)]


class agentBase(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState: GameState) -> None:
        """
        Required.

        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        self.start = gameState.getAgentPosition(self.index)
        # the following initialises self.red and self.distancer
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState: GameState) -> Directions:
        """
        Required.

        This is called each turn to get an agent to choose and action

        Return:
        This find the directions by going through gameState.getLegalAction
        - don't try and generate this by manually
        """
        actions = gameState.getLegalActions(self.index)

        return random.choice(actions)


stepListRecord = []
class offensiveAgent(agentBase):
    # def __init__(self, index, timeForComputing=.1):
    #     super().__init__(index, timeForComputing)
    #     self.stepsSinceLastScore = 0  # 跟踪步骤数的属性

    # def registerInitialState(self, gameState: GameState) -> None:
    #     super().registerInitialState(gameState)
    #     self.stepsSinceLastScore = 0  # 重置步骤计数

    def chooseAction(self, gameState: GameState) -> Directions:
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action

        problem = FoodOffenseWithAgentAwareness(
            startingGameState=gameState, captureAgent=self)
        try:
            with time_limit(1):
                actions = aStarSearch(problem, heuristic=offensiveHeuristic)
            # this can occure if start in the goal state. In this case do not want to perform any action.
            if actions == []:
                actions == ["Stop"]

        except TimeoutException as e:
            print("TimeoutException")
            actions = [random.choice(gameState.getLegalActions(self.index))]

        except SolutionNotFound as e:
            print("NotSolutionFound")
            actions = [random.choice(gameState.getLegalActions(self.index))]

        return actions[0]
    
    def final(self, state):
        global stepListRecord
        # TODO calculate average steps  ....
        # print("Total scores steps: ", stepListRecord)


#################  problems and heuristics  ####################

def uniform_agent_direction(gameState):
    '''
    the agent direction is considered when checking for equality of game state.
    This is not important to us and creates more states than required, so set them all to be constant
    '''
    default_direction = Directions.NORTH

    for agent_state in gameState.data.agentStates:
        if agent_state.configuration:
            agent_state.configuration.direction = default_direction
        else:
            pass  # this happens when non enemy agent is visible - not required to do anything here

    return gameState


class FoodOffenseWithAgentAwareness():
    '''
    This problem extends FoodOffense by updateing the enemy ghost to move to our pacman if they are adjacent (basic Goal Recognition techniques).
    This conveys to our pacman the likely effect of moving next to an enemy ghost - but doesn't prohibit it from doing so (e.g if Pacman has been trapped)

    Note: This is a SearchProblem class. It could inherit from search.Search problem (mainly for conceptual clarity).
    '''

    def __init__(self, startingGameState: GameState, captureAgent: CaptureAgent):
        self.expanded = 0
        self.startingGameState = uniform_agent_direction(startingGameState)
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.scoreChange = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 1
        # WARNING: Capture agent doesn't update with new state, this should only be used for non state dependant utils (e.g distancer)
        self.captureAgent: CaptureAgent = captureAgent
        self.goal_state_found = None

    def getStartState(self):
        # This needs to return the state information to being with
        return (self.startingGameState, self.startingGameState.getScore(), True)

    def isGoalState(self, state: t.Tuple[GameState]) -> bool:
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount

        # Note: can't use CaptureAgent, as it doesn't update with game state
        gameState = state[0]

        # If red team, want scores to go up
        if self.captureAgent.red == True:
            if gameState.data.scoreChange >= self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                False
        # If blue team, want scores to go down
        else:
            if gameState.data.scoreChange <= -self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                False

    def getSuccessors(self, state: t.Tuple[GameState], node_info: t.Optional[dict] = None) -> t.List[t.Tuple[t.Tuple[GameState, bool], Directions, int]]:
        gameState = state[0]

        actions = gameState.getLegalActions(self.captureAgent.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        next_game_states = [gameState.generateSuccessor(self.captureAgent.index, action) for action in actions]

        successors = []
        for action, next_game_state in zip(actions, next_game_states):
            isSafe = True  

            current_depth_of_search = len(node_info["action_from_init"])
            if current_depth_of_search <= self.DEPTH_CUTOFF and gameState.getAgentState(self.captureAgent.index).isPacman:
                self.expanded += 1

                current_agent_index = self.captureAgent.index
                enemy_indexes = next_game_state.getBlueTeamIndices() if next_game_state.isOnRedTeam(current_agent_index) else next_game_state.getRedTeamIndices()

                close_enemy_indexes = [enemy_index for enemy_index in enemy_indexes if next_game_state.getAgentPosition(enemy_index) is not None]
                distancer = self.captureAgent.distancer
                my_pos = next_game_state.getAgentState(current_agent_index).getPosition()
                adjacent_enemy_indexs = list(filter(lambda x: distancer.getDistance(my_pos, next_game_state.getAgentState(x).getPosition()) <= 1, close_enemy_indexes))

                adjacent_ghost_indexs = list(filter(lambda x: (not next_game_state.getAgentState(x).isPacman) and (next_game_state.getAgentState(x).scaredTimer <= 0), adjacent_enemy_indexs))

                ghost_kill_directions = []
                for index in adjacent_ghost_indexs:
                    position = next_game_state.getAgentState(index).getPosition()
                    for action in Actions._directions.keys():
                        new_pos = Actions.getSuccessor(position, action)
                        if new_pos == my_pos:
                            ghost_kill_directions.append(action)
                            isSafe = False 
                            break

                for enemy_index, direction in zip(adjacent_ghost_indexs, ghost_kill_directions):
                    self.expanded += 1
                    next_game_state = next_game_state.generateSuccessor(enemy_index, direction)

            successors.append(((uniform_agent_direction(next_game_state), isSafe), action, 1))

        return successors



# helpers
direction_map = {Directions.NORTH: (0, 1),
                 Directions.SOUTH: (0, -1),
                 Directions.EAST:  (1, 0),
                 Directions.WEST:  (-1, 0),
                 Directions.STOP:  (0, 0)}


def offensiveHeuristic(state, problem=None):
    captureAgent = problem.captureAgent
    index = captureAgent.index
    gameState = state[0]
    isSafe = state[-1]
    # print(isSafe)
    # if getCapsule(captureAgent, gameState)[0] is not None:
    #     capsuleLocation = getCapsule(captureAgent, gameState)[0]

    # check if we have reached a goal state and explicitly return 0
    if captureAgent.red == True:
        if gameState.data.scoreChange >= problem.MINIMUM_IMPROVEMENT:
            return 0
    # If blue team, want scores to go down
    else:
        if gameState.data.scoreChange <= - problem.MINIMUM_IMPROVEMENT:
            return 0
    # check issafe state
    if not isSafe:
        return float('inf')

    agent_state = gameState.getAgentState(index)
    food_carrying = agent_state.numCarrying

    myPos = gameState.getAgentState(index).getPosition()
    distancer = captureAgent.distancer

    # this will be updated to be closest food location if not collect enough food
    return_home_from = myPos

    # still need to collect food
    dist_to_food = 0
    if food_carrying < problem.MINIMUM_IMPROVEMENT:
        # distance to the closest food
        food_list = getFood(captureAgent, gameState).asList()

        min_pos = None
        min_dist = 99999999
        for food in food_list:
            dist = distancer.getDistance(myPos, food)
            if dist < min_dist:
                min_pos = food
                min_dist = dist

        # compare to capsule location
        # if capsuleLocation is not None:
        #     distcap = distancer.getDistance(myPos, capsuleLocation)
        #     if distcap <= min_dist:
        #         dist_to_cap = distcap
        #         return dist_to_cap

        dist_to_food = min_dist
        return_home_from = min_pos
        return dist_to_food

    # Returning Home
    # WARNING: this assumes the maps are always semetrical, territory is divided in half, red on right, blue on left
    walls = list(gameState.getWalls())
    y_len = len(walls[0])
    x_len = len(walls)
    mid_point_index = int(x_len/2)
    if captureAgent.red:
        mid_point_index -= 1

    # find all the entries and find distance to closest
    entry_coords = []
    for i, row in enumerate(walls[mid_point_index]):
        if row is False:  # there is not a wall
            entry_coords.append((int(mid_point_index), int(i)))

    minDistance = min([distancer.getDistance(return_home_from, entry)
                       for entry in entry_coords])
    return dist_to_food + minDistance


# methods required for above heuristic
def getFood(agent, gameState):
    """
    Returns the food you're meant to eat. This is in the form of a matrix
    where m[x][y]=true if there is food you can eat (based on your team) in that square.
    """
    if agent.red:
        return gameState.getBlueFood()
    else:
        return gameState.getRedFood()
    
# new strategy if agent is close to  a capsule
# def getCapsule(agent, gameState):
#     if agent.red:
#         redCapsuleL =  gameState.getRedCapsules()
#         return redCapsuleL
#     else:
#         blueCapsuleL = gameState.getBlueCapsules()
#         return blueCapsuleL


################# Defensive problems and heuristics  ####################


class defensiveAgent(agentBase):

    prevMissingFoodLocation = None
    enemyEntered = False
    boundaryGoalPosition = None

    def chooseAction(self, gameState: GameState):

        problem = defendTerritoryProblem(
            startingGameState=gameState, captureAgent=self)
        actions = aStarSearchDefensive(problem, heuristic=defensiveHeuristic)
        if len(actions) != 0:
            return actions[0]
        else:
            return random.choice(gameState.getLegalActions(self.index))
            # return 'Stop'
        # return actions[0]


class defendTerritoryProblem():
    def __init__(self, startingGameState: GameState, captureAgent: CaptureAgent):
        self.expanded = 0
        self.startingGameState = startingGameState
        self.captureAgent: CaptureAgent = captureAgent
        self.enemies = self.captureAgent.getOpponents(startingGameState)
        self.walls = startingGameState.getWalls()
        self.intialPosition = self.startingGameState.getAgentPosition(
            self.captureAgent.index)
        self.gridWidth = self.captureAgent.getFood(startingGameState).width
        self.gridHeight = self.captureAgent.getFood(startingGameState).height
        # red team on the left  blue team on the right
        if self.captureAgent.red:
            self.boundary = int(self.gridWidth / 2) - 1
            self.myPreciousFood = self.startingGameState.getRedFood()
        else:
            self.boundary = int(self.gridWidth / 2)
            self.myPreciousFood = self.startingGameState.getBlueFood()

        (self.viableBoundaryPositions,
         self.possibleEnemyEntryPositions) = self.getViableBoundaryPositions()

        self.GOAL_POSITION = self.getGoalPosition()
        self.goalDistance = self.captureAgent.getMazeDistance(
            self.GOAL_POSITION, self.intialPosition)

    def getViableBoundaryPositions(self):
        myPos = self.startingGameState.getAgentPosition(
            self.captureAgent.index)
        b = self.boundary
        boundaryPositions = []
        enemyEntryPositions = []

        for h in range(0, self.gridHeight):
            if self.captureAgent.red:
                if not(self.walls[b][h]) and not(self.walls[b+1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b+1, h))

            else:
                if not(self.walls[b][h]) and not(self.walls[b-1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b-1, h))

        return (boundaryPositions, enemyEntryPositions)

    def getGoalPosition(self):
        isPacman = self.startingGameState.getAgentState(
            self.captureAgent.index).isPacman

        isScared = self.startingGameState.getAgentState(
            self.captureAgent.index).scaredTimer > 0

        if isScared:
            boundaryGoalPositions = self.closestPosition(
                self.intialPosition, self.viableBoundaryPositions)
            if self.captureAgent.boundaryGoalPosition == None:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                    boundaryGoalPosition = boundaryGoalPositions.pop()
                    self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
                else:
                    boundaryGoalPosition = self.captureAgent.boundaryGoalPosition
            return boundaryGoalPosition

        missingFoodPosition = self.getMissingFoodPosition()

        if missingFoodPosition != None:
            self.captureAgent.prevMissingFoodLocation = missingFoodPosition
            return missingFoodPosition

        for enemy in self.enemies:
            if self.startingGameState.getAgentState(enemy).isPacman:
                self.captureAgent.enemyEntered = True
                if self.startingGameState.getAgentPosition(enemy) != None:
                    return self.startingGameState.getAgentPosition(enemy)
                else:
                    return self.getProbableEnemyEntryPointBasedOnFood()
            else:
                self.captureAgent.enemyEntered = False

        if self.captureAgent.prevMissingFoodLocation != None and self.captureAgent.enemyEntered:
            return self.captureAgent.prevMissingFoodLocation

        # TODO: tidy this code as it is clarified before
        boundaryGoalPositions = self.closestPosition(
            self.intialPosition, self.viableBoundaryPositions)

        if self.captureAgent.boundaryGoalPosition == None:
            boundaryGoalPosition = boundaryGoalPositions.pop()
            self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
        else:
            if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                boundaryGoalPosition = self.captureAgent.boundaryGoalPosition

        return boundaryGoalPosition

    def closestPosition(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        for toPos in positions:
            positionsSorted.push(
                toPos, self.captureAgent.getMazeDistance(toPos, fromPos))
        return positionsSorted

    def getProbableEnemyEntryPointBasedOnFood(self):
        positionsSorted = util.PriorityQueue()
        bestEnemyPosition = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.intialPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            # getDistanceOnGrid(self.intialPosition, possibleEntry) is used for enhanced patrol
            if self.captureAgent.distancer.getDistanceOnGrid(self.intialPosition, possibleEntry) > 2:
                closestFoodPosition = self.closestPosition(
                    possibleEntry, self.myPreciousFood.asList()).pop()
                distancetoToClosestFoodFromPosition = self.captureAgent.getMazeDistance(
                    possibleEntry, closestFoodPosition)
                bestEnemyPosition.push(
                    possibleEntry, distancetoToClosestFoodFromPosition)

        bestEnemyEntryPosition = bestEnemyPosition.pop()

        if bestEnemyEntryPosition:
            return bestEnemyEntryPosition
        else:
            return random.choice(self.possibleEnemyEntryPositions)

    def getMissingFoodPosition(self):

        prevFood = self.captureAgent.getFoodYouAreDefending(self.captureAgent.getPreviousObservation()).asList() \
            if self.captureAgent.getPreviousObservation() is not None else list()
        currFood = self.captureAgent.getFoodYouAreDefending(
            self.startingGameState).asList()

        if prevFood:
            if len(prevFood) > len(currFood):
                foodEaten = list(set(prevFood) - set(currFood))
                if foodEaten:
                    return foodEaten[0]
        return None

    def getStartState(self):
        return (self.startingGameState, self.goalDistance)

    def isGoalState(self, state: (GameState, int)): # type: ignore

        gameState = state[0]

        (x, y) = myPos = gameState.getAgentPosition(self.captureAgent.index)

        if myPos == self.GOAL_POSITION:
            return True
        else:
            return False

    def getSuccessors(self, state: (GameState, int), node_info=None): # type: ignore
        self.expanded += 1

        gameState = state[0]

        actions: t.List[Directions] = gameState.getLegalActions(
            self.captureAgent.index)

        goalDistance = self.captureAgent.getMazeDistance(
            self.GOAL_POSITION, gameState.getAgentPosition(self.captureAgent.index))

        successors_all = [((gameState.generateSuccessor(
            self.captureAgent.index, action), goalDistance), action, 1) for action in actions]

        successors = []

        for successor in successors_all:
            (xs, ys) = successor[0][0].getAgentPosition(
                self.captureAgent.index)
            if self.captureAgent.red:
                if xs <= self.boundary:
                    successors.append(successor)
            else:
                if xs >= self.boundary:
                    successors.append(successor)

        return successors

    def getCostOfActions(self, actions):
        util.raiseNotDefined()


def defensiveHeuristic(state, problem=None):
    gameState = state[0]
    currGoalDistance = state[1]

    succGoalDistance = problem.captureAgent.getMazeDistance(
        problem.GOAL_POSITION, gameState.getAgentPosition(problem.captureAgent.index))

    if succGoalDistance < currGoalDistance:
        return 0
    else:
        return float('inf')



class TimeoutException(Exception):
    pass


# @contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise TimeoutException("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

# @contextmanager
# def time_limit(seconds):
#     timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutException("Timed out!")))
#     timer.start()
#     try:
#         yield
#     finally:
#         timer.cancel()

@contextmanager
def time_limit(seconds):
    timeout_event = threading.Event()

    def trigger_timeout():
        timeout_event.set()

    timer = threading.Timer(seconds, trigger_timeout)
    timer.start()

    try:
        yield timeout_event
    finally:
        timer.cancel()


################# Search Algorithems ###################


class SolutionNotFound(Exception):
    pass


class Node():
    def __init__(self, *, name, steps_from_start=0):
        self.name = name
        self.steps_from_start = steps_from_start

    def add_info(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    global stepListRecord
    node = Node(name="n0").add_info(state=problem.getStartState())
    h_n = heuristic(node.state, problem=problem)
    g_n = 0  # accumulated cost so far
    node.add_info(
        f_n=g_n + h_n,  # f unction to sort the priority queue by
        g_n=g_n,  # accumilated cost so far
        action_from_init=[],
    )

    op = util.PriorityQueue()
    op.push(node, priority=node.f_n)
    close = set()  # state based
    best_g = {}  # key = state, value = g_n

    # total_expanded = 0
    # reopen_count = 0

    while not op.isEmpty():
        count = 1
        node = op.pop()
        if (node.state not in close) or (node.g_n < best_g[node.state]):
            # total_expanded += 1
            # print("------------")
            # # print(node.state[0])
            # print(len(node.action_from_init))
            # print(f"f_n {node.f_n}")
            # print(f"g_n {node.g_n}")
            # print(f"Nodes expanded {total_expanded}")
            # if node.state in close:
            #     print(f"node reopened")
            #     reopen_count +=1
            #     print(f"total reopens {reopen_count}")
            #     print("previous g_n improved on {best_g[node.state]}")
            # print("------------")
            close.add(node.state)
            best_g[node.state] = node.g_n
            if problem.isGoalState(node.state):
                global stepListRecord
                stepListRecord.append(len(node.action_from_init))
                break
            else:
                for related_node in problem.getSuccessors(node.state, node_info={"action_from_init": [*node.action_from_init]}):
                    new_state, action, step_cost = related_node[0], related_node[1], related_node[2]
                    g_n = node.g_n + step_cost
                    h_n = heuristic(new_state, problem=problem)
                    if h_n < float('inf'):  # solution is possible
                        new_node = Node(name=f"n{count}", steps_from_start=node.steps_from_start + 1).add_info(
                            state=new_state,
                            f_n=g_n + h_n,
                            g_n=g_n,
                            # important step to keep on track of the taken action
                            action_from_init=[*node.action_from_init] + [action],
                        )
                        # print(new_state[0].data.scoreChange)
                        # if new_state[0].getScore() - node.state[0].getScore() >= 2:
                        #     print(new_state[0].getScore(), node.state[0].getScore())
                        #     print(f"Steps taken to score: {new_node.steps_from_start}")
                        # checking if goal here improves performance
                        # also protects agains bad heuristics that would send a goal to the end of the heap
                        if problem.isGoalState(node.state):
                            node = new_node
                            break
                        count += 1
                        op.push(new_node, new_node.f_n)
    else:
        raise SolutionNotFound({"start_state": problem.getStartState})

    # debugging info will be left for future users
    # print("--------------- FINAL RESULTS -----------------")
    # print(node.action_from_init)
    # print(len(node.action_from_init))
    # print(f"node reopened {reopen_count}")
    return node.action_from_init


def aStarSearchDefensive(problem, heuristic=nullHeuristic):
    explore = util.PriorityQueue()

    initial_state = problem.getStartState()
    h = heuristic(initial_state, problem)
    g = 0

    explore.push((initial_state, [], g), g + h)

    visited_states = []
    best_g = 0

    while not explore.isEmpty():
        state, action, g = explore.pop()

        if state not in visited_states or g < best_g:
            visited_states.append(state)

            if g < best_g:
                best_g = g

            if problem.isGoalState(state):
                return action

            for child_state, child_action, child_cost in problem.getSuccessors(state):
                child_heuristic = heuristic(child_state, problem)

                if child_heuristic < float("inf"):
                    explore.push(
                        (child_state, action + [child_action], g + child_cost),
                        g + child_cost + child_heuristic,
                    )

    return []
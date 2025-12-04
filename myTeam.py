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
import numpy as np
import torch
from torch import nn

from captureAgents import CaptureAgent
import random

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


"""
Todo:
- 
- Rename this file to TEAM_NAME.py 
- Load in your model which should start with your TEAM_NAME
"""

#####################
# Agent model class #
#####################

class AgentQNetwork(nn.Module):
    def __init__(self):
        super(AgentQNetwork, self).__init__()

    def forward(self, obs):

        return q_values

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Agent1', second='Agent3', dir=''):
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
    return [eval(first)(firstIndex, dir), eval(second)(secondIndex, dir)]


##########
# Agents #
##########

class Agent1(CaptureAgent):
    def __init__(self, index, dir=''):
        super().__init__(index)
        # The dir variable is only used for the competition so you can ignore that
        # Todo: change model name
        self.Q = AgentQNetwork().to(device)
        self.Q.load_state_dict(torch.load(dir + "MODEL_NAME.pth"))

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
        state = get_Observation(self.index, gameState).to(device)
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(device)
        # Todo: flatten if required
        if self.index in [1,3]: # If team blue normal mapping
            action_mapping = {
                0: "North",
                1: "East",
                2: "South",
                3: "West",
                4: "Stop"
            }
        else: # If team red the actions have to be mirrored as the observation is mirrored
            action_mapping = {
                0: "North",
                3: "East",
                2: "South",
                1: "West",
                4: "Stop"
            }
        # Exploit: take the action with the highest Q-value
        q_values = self.Q(state).cpu().detach().numpy()[0]
        index = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        action = action_mapping[index]

        return action

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


class Agent3(CaptureAgent):
    def __init__(self, index, dir=''):
        # The dir variable is only used for the competition so you can ignore that
        # Todo: change model name
        super().__init__(index)
        self.Q = AgentQNetwork().to(device)
        self.Q.load_state_dict(torch.load(dir + "MODEL_NAME.pth"))

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
        state = get_Observation(self.index, gameState).to(device)
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(device)
        # Todo: flatten if required

        if self.index in [1,3]: # If team blue normal mapping
            action_mapping = {
                0: "North",
                1: "East",
                2: "South",
                3: "West",
                4: "Stop"
            }
        else: # If team red the actions have to be mirrored as the observation is mirrored
            action_mapping = {
                0: "North",
                3: "East",
                2: "South",
                1: "West",
                4: "Stop"
            }
        # Exploit: take the action with the highest Q-value
        q_values = self.Q(state).cpu().detach().numpy()[0]
        index = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        action = action_mapping[index]

        return action

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


def get_Observation(agentIndex, state):
    locations = []
    for i in range(4):
        if state.getAgentPosition(i):
            locations.append(state.getAgentPosition(i)[::-1])
        else:
            locations.append(None)

    # locations = [state.getAgentPosition(i)[::-1] for i in range(4)]
    layout_shape = (state.data.layout.height, state.data.layout.width)
    location = torch.zeros(layout_shape)
    alliesLocations = torch.zeros(layout_shape)
    enemiesLocations = torch.zeros(layout_shape)
    blueCapsules = torch.zeros(layout_shape)
    redCapsules = torch.zeros(layout_shape)

    observation = torch.unsqueeze(torch.tensor(state.getWalls().data).T, dim=0)
    if locations[agentIndex]:
        location[locations[agentIndex]] = 1 + state.getAgentState(agentIndex).numCarrying
    observation = torch.cat((observation, location.unsqueeze(0)), dim=0)

    for otherAgent in range(4):
        if otherAgent != agentIndex:
            if (otherAgent in [0,2] and agentIndex in [0,2]) or (otherAgent in [1,3] and agentIndex in [1,3]):
                if locations[agentIndex]:
                    alliesLocations[locations[otherAgent]] = 1
                for capX, capY in state.getBlueCapsules():
                    blueCapsules[capY, capX] = 1
            else:
                if locations[agentIndex]:
                    enemiesLocations[locations[otherAgent]] = 1
                for capX, capY in state.getRedCapsules():
                    redCapsules[capY, capX] = 1

    if agentIndex in [1,3]:
        # Observation remains unchanged for the blue team
        observation = torch.cat((
            observation,
            blueCapsules.unsqueeze(0),
            redCapsules.unsqueeze(0),
            alliesLocations.unsqueeze(0),
            enemiesLocations.unsqueeze(0),
            torch.unsqueeze(torch.tensor(state.getBlueFood().data).T, dim=0),
            torch.unsqueeze(torch.tensor(state.getRedFood().data).T, dim=0)
        ), dim=0)
    else:
        # Create a completely new observation for the red team
        flipped_observation = torch.flip(observation, dims=[2])  # Flip observation horizontally
        flipped_blue_capsules = torch.flip(redCapsules.unsqueeze(0), dims=[2])  # Flip red capsules as blue
        flipped_red_capsules = torch.flip(blueCapsules.unsqueeze(0), dims=[2])  # Flip blue capsules as red
        flipped_allies = torch.flip(alliesLocations.unsqueeze(0), dims=[2])  # Flip ally locations
        flipped_enemies = torch.flip(enemiesLocations.unsqueeze(0), dims=[2])  # Flip enemy locations
        flipped_blue_food = torch.flip(torch.unsqueeze(torch.tensor(state.getRedFood().data).T, dim=0),
                                       dims=[2])  # Red food as blue
        flipped_red_food = torch.flip(torch.unsqueeze(torch.tensor(state.getBlueFood().data).T, dim=0),
                                      dims=[2])  # Blue food as red

        # Combine flipped components into the final observation
        observation = torch.cat((
            flipped_observation,
            flipped_blue_capsules,
            flipped_red_capsules,
            flipped_allies,
            flipped_enemies,
            flipped_blue_food,
            flipped_red_food
        ), dim=0)

    return observation
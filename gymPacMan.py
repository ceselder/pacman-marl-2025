import functools
import os.path
import numpy as np
import torch

import layout
from capture import CaptureRules, loadAgents, randomLayout, GameState, AgentRules
from captureAgents import CaptureAgent

file_path = os.path.dirname(os.path.abspath(__file__)) + '/'

class gymPacMan_parallel_env:
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, layout_file = f'{file_path}/layouts/smallCapture.lay', display = False, length = 299, reward_forLegalAction= True, defenceReward = True, random_layout = False, enemieName = 'randomTeam', self_play = False):
        if os.path.exists(layout_file) and not random_layout:
            l = layout.getLayout(layout_file)
            print("Loaded layout from file")
        else:
            l = layout.Layout(randomLayout().split('\n'))
            print("Loaded random layout")
        self.random_layout = random_layout
        self.enemieName =  enemieName
        self.self_play = self_play
        self.layout = l
        self.agents = []
        self.rules = CaptureRules()
        self.display = display
        self.display_bool = display
        self.reward_forLegalAction = reward_forLegalAction
        for agent in loadAgents(True, f'{file_path}/agents/{self.enemieName}', None, ''):
            self.agents.append(agent)
        new_agents = [None for i in range(4)]
        if not self.self_play:
            for i in range(2):
                new_agents[self.agents[i].index] = self.agents[i]
        for i in range(4):
            if new_agents[i] is None:
                new_agents[i] = i
        self.agents = new_agents
        if display:
            import captureGraphicsDisplay
            # Hack for agents writing to the display
            captureGraphicsDisplay.FRAME_TIME = 0
            display =  captureGraphicsDisplay.PacmanGraphics('Random Agents', 'Our Team', 1, 0,
                                                             capture=True)
            import __main__
            __main__.__dict__['_display'] = display
            self.display = display
            self.rules.quiet = False
        else:
            import textDisplay
            self.display = textDisplay.NullGraphics()
            self.rules.quiet = True
        self.length = length
        self.num_moves = 0
        self.steps = 0
        self.game = self.rules.newGame(self.layout, self.agents, self.display, self.length, True, False)
        for agent in self.agents:
            if not isinstance(agent, int):
                agent.registerInitialState(self.game.state)
        if self.display:
            self.display.initialize(self.game.state.data)
        self.max_score = np.sum(self.game.state.getBlueFood().data)
        self.defenceReward = defenceReward
        self.action_mapping = {
            0: "North",
            1: "East",
            2: "South",
            3: "West",
            4: "Stop",
            "North": "North",
            "East": "East",
            "South": "South",
            "West": "West",
            "Stop": "Stop",
        }
        self.reversed_action_mapping = {
            "North": 0,
            "East": 1,
            "South": 2,
            "West": 3,
            "Stop": 4,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }


    def reset(self, layout_file='None', enemieName = 'None'):
        observations = {}
        self.steps = 0
        if os.path.exists(layout_file):
            self.layout = layout.getLayout(layout_file)
            print("Loaded layout from file")
        if self.random_layout:
            self.layout = layout.Layout(randomLayout().split('\n'))
            print("Loaded random layout")
        if enemieName != 'None':
            self.enemieName = enemieName
            self.agents = []
            for agent in loadAgents(True, f'{file_path}/agents/{self.enemieName}', None, ''):
                self.agents.append(agent)
        new_agents = [None for i in range(4)]
        if not self.self_play:
            for i in [0,1] if enemieName != 'None' else [0,2]:
                new_agents[self.agents[i].index] = self.agents[i]
        for i in range(4):
            if new_agents[i] is None:
                new_agents[i] = i
        self.agents = new_agents
        if self.display_bool:
            import captureGraphicsDisplay
            # Hack for agents writing to the display
            captureGraphicsDisplay.FRAME_TIME = 0
            display = captureGraphicsDisplay.PacmanGraphics('Random Agents', 'Our Team', 1, 0,
                                                            capture=True)
            import __main__
            __main__.__dict__['_display'] = display
            self.display = display
            self.rules.quiet = False
        else:
            import textDisplay
            self.display = textDisplay.NullGraphics()
            self.rules.quiet = True
        self.num_moves = 0
        self.game = self.rules.newGame(self.layout, self.agents, self.display, self.length, True, False)
        for agent in self.agents:
            if not isinstance(agent, int):
                agent.registerInitialState(self.game.state)
        if self.display:
            self.display.initialize(self.game.state.data)
        for agentIndex in range(len(self.agents)):
            observation = self.get_Observation(agentIndex)
            observations[self.agents[agentIndex]] = observation
        return observations, {
            'legal_actions': {
                self.agents[x]: [self.reversed_action_mapping[y] for y in self.game.state.getLegalActions(x)] for x in
                range(len(self.agents))}}

    def step(self, actions):
        rewards = {}
        observations = {}
        capture, score_change = 0,0
        blue_reward, red_reward = 0,0
        blue_score_change, red_score_change = 0,0
        for agentIndex in range(len(self.agents)):
            if agentIndex in [1,3]:
                blue_reward = self.get_reward(agentIndex, self.action_mapping[actions[self.agents[agentIndex]]], blue_reward)
                self.game.state = self.game.state.generateSuccessor(agentIndex, self.action_mapping[actions[self.agents[agentIndex]]])
                score_change += self.game.state.data.scoreChange
                blue_score_change -= self.game.state.data.scoreChange
            else:
                if self.self_play: action = self.action_mapping[actions[self.agents[agentIndex]]]
                else: action = self.agents[agentIndex].getAction(self.game.state)
                red_reward = self.get_reward(agentIndex, action, red_reward)
                self.game.state = self.game.state.generateSuccessor(agentIndex,action)
                score_change += self.game.state.data.scoreChange
                #red_score_change = self.game.state.data.scoreChange
                red_score_change += self.game.state.data.scoreChange



            observation = self.get_Observation(agentIndex)
            observations[self.agents[agentIndex]] = observation
            if self.display:
                self.display.update(self.game.state.data)

        blue_reward = blue_reward + max(blue_score_change, 0)
        red_reward =  red_reward + max(red_score_change, 0)

        terminations = self.check_termination()

        # Win/lose bonus - only on game end!
        if any(terminations.values()):
            print(f"end score {self.game.state.data.score}")
            final_score = self.game.state.data.score
            if final_score > 0:
                blue_reward += 10.0 + (1/10) * abs(final_score)
            elif final_score < 0:
                red_reward += 10.0 + (1/10) * abs(final_score)

        for agentIndex in range(len(self.agents)):
            if agentIndex in [0,2]:
                rewards[self.agents[agentIndex]] = red_reward
            else:
                rewards[self.agents[agentIndex]] = blue_reward
        self.steps += 1

        return observations, rewards, terminations, {'legal_actions': {
                self.agents[x]: [self.reversed_action_mapping[y] for y in self.game.state.getLegalActions(x)] for x in
                range(len(self.agents))}, "score_change": score_change}

    def get_Observation(self, agentIndex):
        locations = [self.game.state.getAgentPosition(i)[::-1] for i in range(len(self.agents))]
        layout_shape = (self.layout.height, self.layout.width)
        location = torch.zeros(layout_shape)
        alliesLocations = torch.zeros(layout_shape)
        enemiesLocations = torch.zeros(layout_shape)
        blueCapsules = torch.zeros(layout_shape)
        redCapsules = torch.zeros(layout_shape)

        observation = torch.unsqueeze(torch.tensor(self.game.state.getWalls().data).T, dim=0)
        location[locations[agentIndex]] = 1 + self.game.state.getAgentState(agentIndex).numCarrying
        observation = torch.cat((observation, location.unsqueeze(0)), dim=0)

        for otherAgent in range(len(self.agents)):
            if otherAgent != agentIndex:
                if (otherAgent in [0,2] and agentIndex in [0,2]) or (otherAgent in [1,3] and agentIndex in [1,3]):
                    alliesLocations[locations[otherAgent]] = 1
                    for capX, capY in self.game.state.getBlueCapsules():
                        blueCapsules[capY, capX] = 1
                else:
                    enemiesLocations[locations[otherAgent]] = 1
                    for capX, capY in self.game.state.getRedCapsules():
                        redCapsules[capY, capX] = 1

        observation = torch.cat((
            observation,
            blueCapsules.unsqueeze(0),
            redCapsules.unsqueeze(0),
            alliesLocations.unsqueeze(0),
            enemiesLocations.unsqueeze(0),
            torch.unsqueeze(torch.tensor(self.game.state.getBlueFood().data).T, dim=0),
            torch.unsqueeze(torch.tensor(self.game.state.getRedFood().data).T, dim=0)
        ), dim=0)

        return observation

    def get_reward(self, agentIndex, action, reward):
        next = self.game.state.generateSuccessor(agentIndex, action)
        if  agentIndex in [0,2]:
            if np.array(self.game.state.getRedFood().data).sum() < np.array(next.getRedFood().data).sum():
                reward += 1
            if np.array(self.game.state.getBlueFood().data).sum() > np.array(next.getBlueFood().data).sum():
                reward += 0.1
            if self.defenceReward:
                for blueTeamIndex in [1,3]:
                    if self.game.state.data.agentStates[blueTeamIndex].isPacman and not next.data.agentStates[blueTeamIndex].isPacman:
                        if next.data.agentStates[blueTeamIndex].configuration.pos == next.data.agentStates[blueTeamIndex].start.pos:
                            reward += 0.25

        else:
            if np.array(self.game.state.getBlueFood().data).sum() < np.array(next.getBlueFood().data).sum():
                reward += 1
            if np.array(self.game.state.getRedFood().data).sum() > np.array(next.getRedFood().data).sum():
                reward += 0.1
            if self.defenceReward:
                for redTeamIndex in [0,2]:
                    if self.game.state.data.agentStates[redTeamIndex].isPacman and not next.data.agentStates[redTeamIndex].isPacman:
                        if next.data.agentStates[redTeamIndex].configuration.pos == next.data.agentStates[redTeamIndex].start.pos:
                            reward += 0.25
        if self.reward_forLegalAction:
            legal = AgentRules.getLegalActions(self.game.state, agentIndex)
            if action in legal:
                reward += 0.01

        return reward

    def check_termination(self):
        if np.sum(self.game.state.getBlueFood().data) == 0:
            if self.game.state.getAgentState(0).numCarrying == 0 and self.game.state.getAgentState(2).numCarrying == 0:
                return {agent: True for agent in self.agents}
        if np.sum(self.game.state.getRedFood().data) == 0:
            if self.game.state.getAgentState(1).numCarrying == 0 and self.game.state.getAgentState(3).numCarrying == 0:
                return {agent: True for agent in self.agents}
        if self.steps >= self.length:
            return {agent: True for agent in self.agents}
        return {agent: False for agent in self.agents}

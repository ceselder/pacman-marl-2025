import functools
import os.path
import numpy as np
import torch

import layout as layout
from capture import CaptureRules, loadAgents, randomLayout, GameState, AgentRules
from captureAgents import CaptureAgent


class gymPacMan_parallel_env:
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, team1, team2, layout_file = 'pls.lay', random_agents=False, display = False, length = 1200, self_play=False):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.team1 = team1
        self.team2 = team2
        if os.path.exists(layout_file):
            l = layout.getLayout(layout_file)
        else:
            l = layout.Layout(randomLayout().split('\n'))
        self.layout = l
        self.random_agents = random_agents
        self.agents = []
        self.rules = CaptureRules()
        self.display = display
        agents_team1 = loadAgents(False, team1, None, {'dir': 'test'})
        agents_team2 = loadAgents(True, team2, None, {'dir': 'test'})
        for i in range(2):
            self.agents.append(agents_team1[i])
            self.agents.append(agents_team2[i])

        if display:
            import captureGraphicsDisplay
            # Hack for agents writing to the display
            captureGraphicsDisplay.FRAME_TIME = 0
            display =  captureGraphicsDisplay.PacmanGraphics('baselineT', 'baselineTeam', 1, 0,
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
        # initState = GameState()
        # initState.initialize(self.layout, len(self.agents))
        self.game = self.rules.newGame(self.layout, self.agents, self.display, self.length, True, False)

        self.max_score = len(self.game.state.getRedFood().data)

        for agent in self.agents:
            if not isinstance(agent,str):
                agent.registerInitialState(self.game.state)
        # self.game.state = initState
        if self.display:
            self.display.initialize(self.game.state.data)

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

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        observations = {}
        self.agents = []

        agents_team1 = loadAgents(False, self.team1, None, '')
        agents_team2 = loadAgents(True, self.team2, None, '')
        for i in range(2):
            self.agents.append(agents_team1[i])
            self.agents.append(agents_team2[i])

        # self.game.state = initState
        if self.display:
            self.display.initialize(self.game.state.data)
        self.game = self.rules.newGame(self.layout, self.agents, self.display, self.length, True, False)
        for agent in self.agents:
            if not isinstance(agent,str):
                agent.registerInitialState(self.game.state)
        return observations, {'legal_actions': {self.agents[x]: self.game.state.getLegalActions(x) for x in range(len(self.agents))}}

    def step(self):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        self.debug = []

        actions= []
        for agent in self.agents:
            actions.append(agent.getAction(self.game.state))

        for agentIndex in range(len(self.agents)):
            action = actions[agentIndex]
            self.game.state = self.game.state.generateSuccessor(self.agents[agentIndex].index,action)
            if self.display:
                self.display.update(self.game.state.data)

        terminations = {agent: False for agent in self.agents} if self.game.state.data.score <= -self.max_score or self.game.state.data.score >= self.max_score else {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= self.length
        return terminations, env_truncation


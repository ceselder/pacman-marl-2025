from os import terminal_size

from evaluate_gym import gymPacMan_parallel_env


def evaluate(team1= 'agents/randomTeam', team2= 'agents/randomTeam', layout= "layouts/bloxCapture.lay"):
    env = gymPacMan_parallel_env(team1, team2, layout, display=True)
    env.reset()
    done = False
    steps = 0
    while not done:
        termination, truncations = env.step()

        if steps >=300:
            done = True
        steps += 1



if __name__ == '__main__':
    evaluate('agents/AstarTeam', 'a', "layouts/bloxCapture.lay")
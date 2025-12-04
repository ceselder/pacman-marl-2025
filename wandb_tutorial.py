import numpy as np
import wandb


def train():
    wandb.init(project='project name', id="optional name of the run")
    # Exmapl: wandb.init(project="reinforcementLabo", id = f"{name_experiment}__{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    for episode in range(100):
        for step in range(300):
            loss = np.random.rand()
            wandb.log({'loss': loss})
        reward = np.random.randint(1,10)
        score = np.random.randint(1, 10)
        wandb.log({'reward': reward})
        wandb.log({'score': score})
        # You can add additional metrics

if __name__ == '__main__':
    wandb.login(key='your api key')
    train()


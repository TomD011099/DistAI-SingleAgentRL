import ray
import json
import gym
import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from dqn import DQNTrainer, DQNModel

if __name__ == "__main__":
    folder = "/home/tom/ray_results/DQNAlgorithm/DQNAlgorithm_CartPole-v1_b4737_00000_0_2020-10-24_13-11-46"

    env_name = "CartPole-v1"
    checkpoint = 572
    num_episodes = 10       # Was 1, amount of episodes to run

    # Def env
    env = gym.make(env_name)
    print(folder + "/params.json")

    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    # Load config
    with open(folder + "/params.json") as json_file:
        config = json.load(json_file)
    trainer = DQNTrainer(env=env_name,
                         config=config)
    # Restore checkpoint
    trainer.restore(folder + "/checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint))

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            step += 1
            env.render()
            print(observation)
            action, _, _ = trainer.get_policy().compute_actions([observation], [])
            print(action)
            observation, reward, done, info = env.step(action[0])
            total_reward += reward
        print("episode {} received reward {} after {} steps".format(episode, total_reward, step))
        avg_reward += total_reward
    print('avg reward after {} episodes {}'.format(avg_reward / num_episodes, num_episodes))
    env.close()
    del trainer

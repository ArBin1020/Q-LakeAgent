import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class FrozenLakeAgent:
    def __init__(self, map_name="8x8", is_slippery=True, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay_rate=0.0001):
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        
        self.rng = np.random.default_rng()

    def train(self, episodes, render=False):
        if render:
            self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery, render_mode='human')
        else:
            self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery)
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))    
        rewards_per_episode = np.zeros(episodes)
        
        for i in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if self.rng.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                state = new_state

            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            if self.epsilon == 0:
                self.alpha = 0.0001

            if reward == 1:
                rewards_per_episode[i] = 1

        self.env.close()
        self.plot_rewards(rewards_per_episode, episodes)
        self.save_q_table("result.pkl")

    def plot_rewards(self, rewards, episodes):
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake8x8.png')

    def save_q_table(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

    def run(self, episodes, is_training=True, render=False):
        if is_training:
            self.train(episodes, render)
        else:
            self.load_q_table("result.pkl")
            self.evaluate(episodes, render)

    def evaluate(self, episodes, render=False):
        if render:
            self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery, render_mode='human')
        else:
            self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery)
            
        rewards_per_episode = np.zeros(episodes)
        
        for i in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = np.argmax(self.q_table[state, :])
                state, reward, terminated, truncated, _ = self.env.step(action)

            if reward == 1:
                rewards_per_episode[i] = 1

        self.env.close()
        self.plot_rewards(rewards_per_episode, episodes)

if __name__ == '__main__':
    agent = FrozenLakeAgent()
    agent.run(10, is_training=False, render=True)
    # agent.run(15000, is_training=True, render=False)

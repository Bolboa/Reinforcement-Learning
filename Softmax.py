import numpy as np
import matplotlib.pyplot as plt


def main():

    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10, 
                        0.25, 0.60, 0.45, 0.75, 0.65]

    N_experiments = 1000  # number of experiments to perform
    N_episodes = 100  # number of episodes per experiment

    class Bandit:
        def __init__(self, bandit_probs):
            self.N = len(bandit_probs)  # number of bandits
            self.prob = bandit_probs
        
        # Get reward (1 for success or 0)
        def get_reward(self, action):
            rand = np.random.random()
            # the larger the probability, the less likely to fail
            reward = 1 if (rand < self.prob[action]) else 0
            return reward

    class Agent:
    
        def __init__(self, bandit):
            # Number of arms
            self.k = np.zeros(bandit.N, dtype=np.int)
            # Step count
            self.n = 1
            # Total mean reward
            self.mean_reward = 0
            # Initialize preferences
            self.Q = np.zeros(bandit.N, dtype=np.float)
                
        def softmax(self):
            self.prob_action = np.exp(self.Q - np.max(self.Q)) / np.sum(np.exp(self.Q - np.max(self.Q)), axis=0)
            
        def update_Q(self, action, actions_not_taken, reward):
            # Update probabilities
            self.softmax()
            
            # Update counts
            self.n += 1
            self.k[action] += 1
            
            # Update total
            self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n
            
            # Update preferences
            self.Q[action] = self.Q[action] + \
                (1./self.k[action]) * (reward - self.mean_reward) * (1 - self.prob_action[action])

            self.Q[actions_not_taken] = self.Q[actions_not_taken] - \
                (1./self.k[action]) * (reward - self.mean_reward) * self.prob_action[actions_not_taken]

        def get_action(self):
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))  # choose random action

    def experiment(agent, bandit, N_episodes):
        reward_history = []
        for i in range(N_episodes):
            action = agent.get_action()
            actions_not_taken = agent.Q != action
            reward = bandit.get_reward(action)
            agent.update_Q(action, actions_not_taken, reward)
            reward_history.append(agent.mean_reward)
        return reward_history
            
    reward_history_avg = np.zeros(N_episodes)

    # Run experiments
    for i in range(N_experiments):        
        bandit = Bandit(bandit_probs)
        agent = Agent(bandit)
        reward_history = experiment(agent, bandit, N_episodes)
        
        # Update long-term averages
        reward_history_avg += (reward_history - reward_history_avg) / (i + 1)

        print(reward_history_avg)

main()
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    # bandit probabilities
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10, 
                        0.25, 0.60, 0.45, 0.75, 0.65]
    
    N_experiments = 100  # number of experiments to perform
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
            self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
            self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated value
            self.prob_action = None

        def update_Q(self, action, reward):
            self.k[action] += 1  # increment the number of times an action is chosen
            # get probability distribution
            self.prob_action = np.exp(self.Q - np.max(self.Q)) / np.sum(np.exp(self.Q - np.max(self.Q)), axis=0)
            # update Q action-value using
            # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a)) * (1 - PR(a))
            self.Q[action] += (1./self.k[action]) * (reward - self.Q[action]) * (1 - self.prob_action[action])


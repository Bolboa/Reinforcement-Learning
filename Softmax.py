import numpy as np
import matplotlib.pyplot as plt


def main():
    
    # bandit probabilities
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10, 
                        0.25, 0.60, 0.45, 0.75, 0.65]

    N_experiments = 100  # number of experiments to perform
    N_episodes = 100  # number of episodes per experiment
    alpha = 0.1  # learning rate


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

        def __init__(self, bandit, alpha):
            self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
            self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated value
            self.mean_reward = 0
            self.n = 1  # step count
            self.prob_action = None
            self.alpha = alpha  # learning rate

        def softmax(self):
            self.prob_action = np.exp(self.Q - np.max(self.Q)) / np.sum(np.exp(self.Q - np.max(self.Q)), axis=0)

        def update_Q(self, action, actions_not_taken, reward):
            self.k[action] += 1  # increment the number of times an action is chosen

            self.mean_reward += (reward - self.mean_reward) / self.n  # mean reward

            # update Q action-value for all actions using
            # Q(a) <- Q(a) + 1/(k+1) * (r(a) - R) * (1 - PR(a))
            self.Q[action] += self.alpha * (reward - self.mean_reward) * (1 - self.prob_action[action])
            # update actions not taken
            self.Q[actions_not_taken] += self.alpha * (reward - self.mean_reward) * (1 - self.prob_action[actions_not_taken])
        
        def get_action(self):
            self.softmax()  # get probability distribution
            return np.random.choice(self.Q, p=self.prob_action)  # choose random action

    def experiment(agent, bandit):
        action_history = []
        reward_history = []
        for episode in range(N_episodes):
            action = agent.get_action()
            actions_not_taken = agent.Q != action
            reward = bandit.get_reward(bandit_probs)
            agent.update_Q(action, actions_not_taken, reward)
            action_history.append(action)
            reward_history.append(reward)
        return (np.array(action_history), np.array(reward_history))            

main()


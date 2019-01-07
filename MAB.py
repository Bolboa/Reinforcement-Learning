import numpy as np
import matplotlib.pyplot as plt


def main():

    # bandit probabilities
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10, 
                        0.25, 0.60, 0.45, 0.75, 0.65]
    
    N_experiments = 100  # number of experiments to perform
    N_episodes = 10000  # number of episodes per experiment
    epsilon = 0.1  # probability of random exploration
    save_fig = True  # if false -> plot, if true save as file in same directory

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

        def __init__(self, bandit, epsilon):
            self.epsilon = epsilon
            self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
            self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated value
        
        # Update Q action-value using:
        # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
        def update_Q(self, action, reward):
            self.k[action] += 1  # increment the number of times an action is chosen
            self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])

        def get_action(self, bandit, force_explore=False):
            rand = np.random.random()  # [0.0, 1.0]
            if (rand < self.epsilon) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit
                return action_explore
            else:
                # choose a random value that is a max value.
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy

    
    def experiment(agent, bandit, N_episodes):
        action_history = []
        reward_history = []
        for episode in range(N_episodes):
            action = agent.get_action(bandit)
            reward = bandit.get_reward(action)
            agent.update_Q(action, reward)
            action_history.append(action)
            reward_history.append(reward)
        return (np.array(action_history), np.array(reward_history))

    N_bandits = len(bandit_probs)
    print("Running multi-armed bandits with N_bandits = {} and agent epsilon = {}".format(N_bandits, epsilon))
    reward_history_avg = np.zeros(N_episodes)  # averaged over every episode
    action_history_sum = ((N_episodes, N_bandits))  # sum action history
    for i in range(N_experiments):
        bandit = Bandit(bandit_probs)  # initialize bandits
        agent = Agent(bandit, epsilon)  # initialize agent
        (action_history, reward_history) = experiment(agent, bandit, N_episodes)  # perform experiment

        if (i + 1) % (N_experiments / 100) == 0:
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("  N_episodes = {}".format(N_episodes))
            print("  bandit choice history = {}".format(
                action_history + 1))
            print("  reward history = {}".format(
                reward_history))
            print("  average reward = {}".format(float(np.sum(reward_history)) / float(len(reward_history))))
            print("")
        # sum up rewards for all experiments
        reward_history_avg += reward_history

    reward_history_avg /= np.float(N_experiments)
    print("reward history avg = {}".format(reward_history_avg))

main()
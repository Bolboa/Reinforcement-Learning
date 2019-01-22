import numpy as np
import matplotlib.pyplot as plt
import random

def main():

    # bandit probabilities
    bandit_probs = [0.50, 0.50, 0.50, 0.50, 0.50, 
                        0.50, 0.50, 0.50, 0.50, 0.50]
    
    N_experiments = 400  # number of experiments to perform
    N_episodes = 200  # number of episodes per experiment
    alpha = 0.1  # step size
    epsilon = 0.1  # probability of random exploration

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

        def update_Q_alpha(self, action, reward, alpha):
            self.k[action] += 1  # increment the number of times an action is chosen
            self.Q[action] += alpha * (reward - self.Q[action])

        def get_action(self, bandit, force_explore=False):
            rand = np.random.random()  # [0.0, 1.0]
            if (rand < self.epsilon) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit
                return action_explore
            else:
                # choose a random value that is a max value.
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy

    # Used for non-stationary
    def random_walk(bandit):
        random_idx = random.randint(0, bandit.N-1)
        random_prob = random.uniform(0, 1)
        new_bandit_probs = bandit_probs
        new_bandit_probs[random_idx] = random_prob
        return Bandit(new_bandit_probs)
    
    def experiment(agent, bandit, N_episodes):
        action_history = []
        reward_history = []
        total_bandits = []
        for episode in range(N_episodes):
            total_bandits.append(bandit)
            action = agent.get_action(bandit)
            reward = bandit.get_reward(action)
            agent.update_Q(action, reward)
            action_history.append(action)
            reward_history.append(reward)
            bandit = random_walk(bandit)
        return (np.array(action_history), np.array(reward_history), total_bandits)
    
    def experiment_total(agent, total_bandits, alpha):
        action_history = []
        reward_history = []
        for bandit in total_bandits:
            action = agent.get_action(bandit)
            reward = bandit.get_reward(action)
            agent.update_Q_alpha(action, reward, alpha)
            action_history.append(action)
            reward_history.append(reward)
        return (np.array(action_history), np.array(reward_history))


    def plot_actions(final_action_history_plot):
        plt.figure(figsize=(18, 12))
        for i in range(N_bandits):
            plt.plot(final_action_history_plot[i])
        plt.show()

    def plot_rewards(final_reward_history, file):
        with open(file, "w") as f:
            for item in final_reward_history:
                f.write("%s\n" % item)

        plt.plot(final_reward_history)
        plt.ylabel("Avg Reward")
        plt.show()


    N_bandits = len(bandit_probs)
    final_reward_history = []
    final_reward_history_alpha_lg = []
    final_reward_history_alpha_sm = []
    final_action_history = np.zeros((N_bandits, N_episodes-1))
    
    # increase training after every set of experiment
    for ep in range(1, N_episodes):
        reward_history_avg = np.zeros(ep)  # averaged over every episode
        reward_history_avg_alpha_lg = np.zeros(ep)
        reward_history_avg_alpha_sm = np.zeros(ep)
        action_history_sum = np.zeros((N_episodes-1, N_bandits))  # sum action history
        # run experiments
        for i in range(N_experiments):
            bandit = Bandit(bandit_probs)  # initialize bandits
            agent = Agent(bandit, epsilon)  # initialize agent
            
            (action_history, reward_history, total_bandits) = experiment(agent, bandit, ep)  # perform experiment
            (action_history_alpha_lg, reward_history_alpha_lg) = experiment_total(agent, total_bandits, 0.1)
            (action_history_alpha_sm, reward_history_alpha_sm) = experiment_total(agent, total_bandits, 0.01)


            # sum up rewards for all experiments
            reward_history_avg += reward_history
            reward_history_avg_alpha_lg += reward_history_alpha_lg
            reward_history_avg_alpha_sm += reward_history_alpha_sm

            # Sum up action history
            for j, (a) in enumerate(action_history):
                action_history_sum[j][a] += 1

        reward_history_avg /= np.float(N_experiments)
        reward_history_avg_alpha_lg /= np.float(N_experiments)
        reward_history_avg_alpha_sm /= np.float(N_experiments)

        final_reward_history.append(float(np.sum(reward_history_avg)) / float(len(reward_history_avg)))
        final_reward_history_alpha_lg.append(float(np.sum(reward_history_avg_alpha_lg)) / float(len(reward_history_avg_alpha_lg)))
        final_reward_history_alpha_sm.append(float(np.sum(reward_history_avg_alpha_sm)) / float(len(reward_history_avg_alpha_sm)))
        
        for b in range(N_bandits):
            mean_action = 100 * action_history_sum[:, b] / N_experiments
            final_action_history[b] += mean_action
        final_action_history /= np.float(ep)
    
    final_action_history_plot = final_action_history / np.float(N_episodes)
    
    # plot optimal action history
    #plot_actions(final_action_history_plot)

    # plot reward history
    plot_rewards(final_reward_history, "incremental_ns_result.txt")
    plot_rewards(final_reward_history_alpha_lg, "incremental_ns_alpha_lg.txt")
    plot_rewards(final_reward_history_alpha_sm, "incremental_ns_alpha_sm.txt")

main()
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def main():

    # bandit probabilities
    bandit_probs = [0.50, 0.50, 0.50, 0.50, 0.50, 
                        0.50, 0.50, 0.50, 0.50, 0.50]

    N_experiments = 100  # number of experiments to perform
    N_episodes = 100  # number of episodes per experiment
    epsilon = 0.1  # probability of random exploration
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
    
        def __init__(self, bandit, epsilon):
            self.epsilon = epsilon
            self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
            self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated values
            self.n = 1  # step count
            self.mean_reward = 0

        # Softmax probabilities      
        def softmax(self):
            self.prob_action = np.exp(self.Q - np.max(self.Q)) / np.sum(np.exp(self.Q - np.max(self.Q)), axis=0)

        # Update Q action-value using:
        # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))  
        def update_Q(self, action, actions_not_taken, reward):
            self.softmax()  # run softmax on estimates
            self.k[action] += 1  # increment the number of times an action is chosen
            
            # update action chosen
            self.Q[action] = self.Q[action] + \
                (1./self.k[action]) * (reward - self.Q[action]) * (1 - self.prob_action[action])

            # update actions not chosen
            self.Q[actions_not_taken] = self.Q[actions_not_taken] - \
                (1./self.k[action]) * (reward - self.Q[action]) * self.prob_action[actions_not_taken]

        def update_Q_mean(self, action, actions_not_taken, reward, alpha=False):
            self.softmax()  # run softmax on estimates
            self.k[action] += 1  # increment the number of times an action is chosen
            self.n += 1  # increment step count

            self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

            if alpha == False:
                # update action chosen
                self.Q[action] = self.Q[action] + \
                    (1./self.k[action]) * (reward - self.mean_reward) * (1 - self.prob_action[action])

                # update actions not chosen
                self.Q[actions_not_taken] = self.Q[actions_not_taken] - \
                    (1./self.k[action]) * (reward - self.mean_reward) * self.prob_action[actions_not_taken]
            else:
                # update action chosen
                self.Q[action] = self.Q[action] + \
                    alpha * (reward - self.mean_reward) * (1 - self.prob_action[action])

                # update actions not chosen
                self.Q[actions_not_taken] = self.Q[actions_not_taken] - \
                    alpha * (reward - self.mean_reward) * self.prob_action[actions_not_taken]

        def get_action(self, bandit, force_explore=False):
            rand = np.random.random()  # [0.0, 1.0]
            if (rand < self.epsilon) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit
                return action_explore
            elfrom random import randintse:
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
        reward_history = []
        action_history = []
        for i in range(N_episodes):
            action = agent.get_action(bandit)
            actions_not_taken = agent.Q != action
            reward = bandit.get_reward(action)
            agent.update_Q_mean(action, actions_not_taken, reward, alpha)
            reward_history.append(reward)
            action_history.append(action)
            bandit = random_walk(bandit)  # random walk (non-stationary)
        return (np.array(action_history), np.array(reward_history))

    def plot_actions(final_action_history_plot):
        plt.figure(figsize=(18, 12))
        print(len(final_action_history_plot))
        for i in range(N_bandits):
            plt.plot(final_action_history_plot[i])
        plt.show()

    def plot_rewards(final_reward_history):
        with open("data/softmax_result.txt", "w") as f:
            for item in final_reward_history:
                f.write("%s\n" % item)

        plt.plot(final_reward_history)
        plt.ylabel("Avg Reward")
        plt.show()

    N_bandits = len(bandit_probs)
    final_reward_history = []
    final_action_history = np.zeros((N_bandits, N_episodes-1))
    
    # increase training after every set of experiment
    for ep in range(1, N_episodes):    
        reward_history_avg = np.zeros(ep)
        action_history_sum = np.zeros((N_episodes-1, N_bandits))  # sum action history
        # run experiments
        for i in range(N_experiments):
            bandit = Bandit(bandit_probs)
            agent = Agent(bandit, epsilon)
            (action_history, reward_history) = experiment(agent, bandit, ep)
                
            # Update long-term averages
            reward_history_avg += reward_history

            # Sum up action history
            for j, (a) in enumerate(action_history):
                action_history_sum[j][a] += 1
            
        reward_history_avg /= np.float(N_experiments)
        final_reward_history.append(float(np.sum(reward_history_avg)) / float(len(reward_history_avg)))

        for b in range(N_bandits):
            mean_action = 100 * action_history_sum[:, b] / N_experiments
            final_action_history[b] += mean_action
        final_action_history /= np.float(ep)
    
    final_action_history_plot = final_action_history / np.float(N_episodes)
    
    # plot optimal action history
    plot_actions(final_action_history_plot)

    # plot reward history
    plot_rewards(final_reward_history)


main()
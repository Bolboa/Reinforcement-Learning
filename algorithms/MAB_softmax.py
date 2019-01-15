import numpy as np
import matplotlib.pyplot as plt

def main():

    # bandit probabilities
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10, 
                        0.25, 0.60, 0.45, 0.75, 0.65]

    N_experiments = 200  # number of experiments to perform
    N_episodes = 200  # number of episodes per experiment
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
            else:
                # choose a random value that is a max value.
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy

    def experiment(agent, bandit, N_episodes):
        reward_history = []
        action_history = []
        for i in range(N_episodes):
            action = agent.get_action(bandit)
            actions_not_taken = agent.Q != action
            reward = bandit.get_reward(action)
            agent.update_Q(action, actions_not_taken, reward)
            reward_history.append(reward)
            action_history.append(action)
        return (np.array(action_history), np.array(reward_history))

    # final_reward_history = []
    # # increase training after every set of experiment
    # for ep in range(1, N_episodes):    
    N_bandits = len(bandit_probs)
    reward_history_avg = np.zeros(N_episodes)
    action_history_sum = np.zeros((N_episodes, N_bandits))  # sum action history
    # Run experiments
    for i in range(N_experiments):
        bandit = Bandit(bandit_probs)
        agent = Agent(bandit, epsilon)
        (action_history, reward_history) = experiment(agent, bandit, N_episodes)
            
        # Update long-term averages
        reward_history_avg += reward_history
        print(reward_history_avg)

        # Sum up action history
        for j, (a) in enumerate(action_history):
            action_history_sum[j][a] += 1

    reward_history_avg /= np.float(N_experiments)
    print(reward_history_avg)


    plt.figure(figsize=(18, 12))
    for i in range(N_bandits):
        action_history_sum_plot = 100 * action_history_sum[:,i] / N_experiments
        plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
                 action_history_sum_plot,
                 linewidth=5.0,
                 label="Bandit #{}".format(i+1))
    plt.show()

    #final_reward_history.append(float(np.sum(reward_history_avg)) / float(len(reward_history_avg)))
    #print("reward history avg = {}".format(reward_history_avg))

    # print(final_reward_history)
    # print(len(final_reward_history))

    # with open("data/softmax_result.txt", "w") as f:
    #     for item in final_reward_history:
    #         f.write("%s\n" % item)

    # plt.plot(final_reward_history)
    # plt.ylabel("Avg Reward")
    # plt.show()

main()
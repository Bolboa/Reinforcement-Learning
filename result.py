import matplotlib.pyplot as plt

def main():
    
    incremental_result = open('data/incremental_result.txt').read().split("\n")
    incremental_result.pop()
    incremental_result = list(map(float, incremental_result))
    
    softmax_result = open('data/softmax_result.txt').read().split("\n")
    softmax_result.pop()
    softmax_result = list(map(float, softmax_result))

    softmax_result_mean = open('data/softmax_result_mean.txt').read().split("\n")
    softmax_result_mean.pop()
    softmax_result_mean = list(map(float, softmax_result_mean))

    softmax_result_mean_alpha = open('data/softmax_result_mean_alpha.txt').read().split("\n")
    softmax_result_mean_alpha.pop()
    softmax_result_mean_alpha = list(map(float, softmax_result_mean_alpha))
    
    plt.plot(incremental_result, label="incremental")
    plt.plot(softmax_result, label="softmax")
    plt.plot(softmax_result_mean, label="softmax mean")
    plt.plot(softmax_result_mean_alpha, label="softmax mean alpha")
    plt.legend(loc="upper left")
    plt.ylabel("Avg Reward")
    plt.show()

main()
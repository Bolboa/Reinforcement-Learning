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

    incremental_ns = open('data/incremental_ns_result.txt').read().split("\n")
    incremental_ns.pop()
    incremental_ns = list(map(float, incremental_ns))

    incremental_ns_alpha_lg = open('data/incremental_ns_alpha_lg.txt').read().split("\n")
    incremental_ns_alpha_lg.pop()
    incremental_ns_alpha_lg = list(map(float, incremental_ns_alpha_lg))

    incremental_ns_alpha_sm = open('data/incremental_ns_alpha_sm.txt').read().split("\n")
    incremental_ns_alpha_sm.pop()
    incremental_ns_alpha_sm = list(map(float, incremental_ns_alpha_sm))
    
    plt.plot(incremental_ns, label="softmax")
    plt.plot(incremental_ns_alpha_lg, label="softmax mean")
    plt.plot(incremental_ns_alpha_sm, label="softmax mean alpha")
    plt.legend(loc="upper left")
    plt.ylabel("Avg Reward")
    plt.show()

main()
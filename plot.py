import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    losses = np.arange(10)
    plt.figure(figsize=(10,6))
    plt.semilogy(losses, c='r', linestyle='-')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("Losses")
    plt.show()

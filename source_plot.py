import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_pressure(x, y):
    plt.plot(x, y, color='tab:blue')
    plt.title("Source Graph for Moment")
    plt.show()

def plot_force(x, y):
    plt.plot(x, y, color='tab:blue')
    plt.title("Source Graph for Force")
    plt.show()

def main():
    args = sys.argv[1:]


if __name__ == '__main__':
    main()

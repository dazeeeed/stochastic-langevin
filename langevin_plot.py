import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import date, datetime
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def readFile(filename):
    return pd.read_csv(str(filename), sep=',',
        header=0) #.values.T[0] # get values -> transpose -> get 1st element

def D1(X):
    return X - X**3

def D2(X):
    return X**2 + 1
 
def langevin(D1, D2, step=0.01, rng=6000):
    step = 0.001
    X = 0
    for i in range(rng):
        X = X + D1(X) * step + np.sqrt(D2(X) * step) * np.random.normal(0, 1)
        yield X

def plot_lanegevin():
    NUMBER = 10000
    step = 1/NUMBER
    rng = 6*NUMBER
    a = list(langevin(D1, D2, step, rng))
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    # ax1.scatter(range(len(a)), a)
    ax1.plot(range(len(a)), a)
    ax1.set_xticks([i*NUMBER for i in range(7)])
    ax1.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax1.set_xlabel('t[s]')
    ax1.set_ylabel('Value')
    ax2.hist(a, bins=200, edgecolor='black')
    ax2.set_ylabel('Counts')
    ax2.set_xlabel('Value')
    plt.show()

def plot_data():
    data = readFile("SP500.csv")

    rows_per_year = 365
    years = int(20 * rows_per_year)

    date_ = pd.to_datetime(data["Date"][-years:])
    open_ = data["Open"][-years:].to_numpy()
    high_ = data["High"][-years:].to_numpy()
    low_ = data["Low"][-years:].to_numpy()
    close_ = data["Close"][-years:].to_numpy()
    adjClose_ = data["Adj Close"][-years:].to_numpy()
    volume_ = data["Volume"][-years:].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    ax1.plot(date_, high_, label='High')
    ax1.plot(date_, low_, label='low')
    ax2.plot(date_, (high_ - low_), label='difference')
    ax3.hist(low_, bins=100, label='Hist of lows')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

def plot_dif():
    data = readFile("SP500.csv")

    rows_per_year = 365
    years = int(1 * rows_per_year)

    date_ = pd.to_datetime(data["Date"][-years:])
    open_ = data["Open"][-years:].to_numpy()

    b = [open_[i+1] - open_[i] for i in range(len(open_) - 1)]
    plt.plot(range(len(b)), b)
    plt.show()

def main():
    # plot_data()
    plot_lanegevin()
    # plot_dif()

if __name__ == '__main__':
    main()
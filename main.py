import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import date, datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def readFile(filename):
    return pd.read_csv(str(filename), sep=',',
        header=0) #.values.T[0] # get values -> transpose -> get 1st element

def calculate_average(x, dt=1, n=1):
    '''Calculate M^(n) (x,t). Returns average as array and size of this array.'''
    arr = x
    max_ = len(x)
    if (len(x)) % dt != 0:
        max_ = (len(x)-1) // dt * dt
        arr = x[:max_]
    m_n = np.array([ (arr[i+1]-arr[i])**n for i in range(max_-1)])
    m_n = np.append(m_n, 0)
    return m_n, max_

def calculate_D(m_n, n, dt):
    return m_n / (np.math.factorial(n)*dt)


if __name__ == "__main__":
    data = readFile("SP500.csv")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    print("Data columns:" + str(data.columns))

    date_ = pd.to_datetime(data["Date"])
    open_ = data["Open"].to_numpy()
    high_ = data["High"].to_numpy()
    low_ = data["Low"].to_numpy()
    close_ = data["Close"].to_numpy()
    adjClose_ = data["Adj Close"].to_numpy()
    volume_ = data["Volume"].to_numpy()

    ax1.plot(date_, open_, date_, high_, date_, low_)

    dt = 1

    m_1, maxsize = calculate_average(open_, dt=dt, n=1)
    m_2, _ = calculate_average(open_, dt=dt, n=2)

    ax2.plot(date_[:maxsize], m_1)
    #ax2.plot(date_[:maxsize], m_2)

    d_1 = calculate_D(m_1, n=1, dt=dt)
    ax2.plot(date_, d_1)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nBye bye!")

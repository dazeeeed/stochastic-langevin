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

# funkcje D1 i D2 do wzorku na X(t)
def D1(X):
    return X - X**3

def D2(X):
    return X**2 + 1
 
def langevin(D1, D2, step=0.01, rng=6000):
    """
    Paramete step okresla gęstosc obliczanych punktów, a rng liczbę obliczanych punktów
    """
    step = 0.001
    X = 0
    for i in range(rng):
        X = X + D1(X) * step + np.sqrt(D2(X) * step) * np.random.normal(0, 1)
        yield X

def plot_lanegevin():
    NUMBER = 10
    step = 1/NUMBER
    rng = 6*NUMBER
    a = list(langevin(D1, D2, step, rng))
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    ax1.scatter(range(len(a)), a)
    # ax1.plot(range(len(a)), a)
    ax1.grid()
    ax1.set_xticks([i*NUMBER for i in range(7)])
    ax1.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax1.set_xlabel('t[s]')
    ax1.set_ylabel('Value')
    ax2.hist(a, bins=5, edgecolor='black')
    ax2.grid()
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

def i_before_k(i, k, lst):
    times = 0
    for i in range(len(lst)-1):
        if lst[i] == i and lst[i+1] == k:
            times += 1
    return times

def occurrences_of_i(i, lst):
    return lst.count(i)

def main():
    # plot_data()
    plot_lanegevin()
    # plot_dif()

if __name__ == '__main__':
    # main()
    # pierwsza lista lst zawiera jakies proste dane do sprawdzenia algorytmu, na nich działa
    # lst = [-1,0,1,2,1,3,2,1,2,1,3,2]
    step = 0.1
    rng = 1000
    # natomiast ta lista zawiera dane wygenerowane przez wzor (2) z pdf z langevina
    lst = list(langevin(D1, D2, step, rng))

    # implementacja binów
    xmin = min(lst)
    xmax = max(lst)
    num_bin = 50
    # lista przechowuje wartosci srodków binów. Liczba binów wynosi num_bin
    bin_center = [xmin + (xmax - xmin)/num_bin * (k+0.5) for k in range(num_bin)]
    bin_width = (xmax - xmin)/num_bin
    half_bin = bin_width / 2

    print("bin centers: ", bin_center)

    # ten slownik przechowuje key:value w taki sposob waartosci: srodka binu:zliczenia w binie
    counts = dict.fromkeys(bin_center, 0)
    # ta lista przechowuje jakby jakie biny wpadają w czasie
    # czyli zamienia wartosci wylosowane w czasie x(t) na biny w czasie
    bins_in_time = list()

    # zliczenia w binach
    # nie wyłapuje skranych wartosci, bo jest problem z zakraglaniem, Do poprawienia.mozna dodac jakis mały epsilon, zeby powiększyc granice
    for value in lst:
        for k in range(num_bin):
            if (bin_center[k] - half_bin) < value <= (bin_center[k] + half_bin):
                bins_in_time.append(bin_center[k])
                counts[bin_center[k]] += 1

    # print(counts)
    print("time: ", bins_in_time)
    # lista m z algorytmu dla odpowienich srodkow binu
    # implementacja algorytmu podesłanego. Oblicza M dla kolejnych środków binu
    m = []

    # poniżej 'i' i 'k' są wartościami środka binu
    for k in bin_center:
        value = 0
        for i in bin_center[::-1]:
            # print(i, " and ", k)
            # tutaj te a jest zawsze zerowe i to jest problemem. 
            # Powinno zwracac liczbe ile razy i wystąpił przed k. Gdzie 'i' i 'k' są srodkami binów. 
            # Czy powinienm rozpatrywac czy to wartosci występują przed sobą. 
            # Jeżeli wartosci, to jakie wartośći z binu rzyjąc do rozpatrywania, czy występują przed sobą?
            a = i_before_k(i, k, bins_in_time)
            b = occurrences_of_i(i, bins_in_time)
            # print("a: ", a)
            # print("b: ", b)
            value += (i - k) *  a / b
        m.append(value)

    print("M: ", m)

    y = [counts[i] for i in counts.keys()]
    plt.scatter(bin_center, y)
    plt.show()
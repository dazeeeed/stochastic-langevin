import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import date, datetime
import sys
from math import factorial, sqrt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def readFile(filename):
    return pd.read_csv(str(filename), sep=',',
        header=0) #.values.T[0] # get values -> transpose -> get 1st element

def D1(X):
    return X - X**3

def D2(X):
    return X**2 + 1

def f1(lst):
    y = [x-x**3 for x in lst]
    return y

def f2(lst):
    y = [x**2+1 for x in lst]
    return y

def plot_poly(lst, poly, order):
    poly_len = len(poly)
    ret_poly = []
    for i in lst:
        value = 0
        for j in range(poly_len):
            value += i**j*poly[poly_len - j - 1]
        ret_poly.append(value)

    return ret_poly
 
def langevin(D1, D2, step=0.001, rng=6000):
    """
    Generation of points given the functions D1(x), D2(x) using stochastic Euler integration.
    Parameters
    ----------
    step - step of integration
    
    rng - number of points to be generated

    Returns
    ----------
    X - one yielded point from integration (one by one)
    """
    X = 0.1
    for i in range(rng):
        X = X + D1(X) * step + np.sqrt(D2(X) * step) * np.random.normal(0, 1)
        yield X

def i_before_k(i, k, lst):
    """Calculate number of i occurences before number k in data series."""
    times = 0
    for j in range(len(lst)-1):
        if lst[j] == i and lst[j+1] == k:
            times += 1
    return times

def occurrences_of_i(i, lst):
    """Calculate number of i occurences in data series."""
    return lst.count(i)

def calculate_D(m, step, order):
    """
    Calculate coefficients given:
    Parameters:
    -----------
    m - list of differences

    order (INT) - power of the coefficient
    
    Returns:
    -----------
    List of coefficient data points.
    """
    return [value/(factorial(order)*step) for value in m]

def calculate_D2(m, step, order):
    return [value * sqrt(2) / (factorial(order)*step) for value in m]

def co_to_jest():
    step = 0.01
    rng = 50000
    # natomiast ta lista zawiera dane wygenerowane przez wzor (2) z pdf z langevina
    lst = list(langevin(D1, D2, step, rng))

    # implementacja binów
    xmin = min(lst)
    xmax = max(lst)
    num_bin = 40
    # lista przechowuje wartosci srodków binów. Liczba binów wynosi num_bin
    bin_center = [xmin + (xmax - xmin)/num_bin * (k+0.5) for k in range(num_bin)]
    bin_width = (xmax - xmin)/num_bin
    half_bin = bin_width / 2
    
    # ten slownik przechowuje key:value w taki sposob waartosci: srodka binu:zliczenia w binie
    counts = dict.fromkeys(bin_center, 0)
    # ta lista przechowuje jakby jakie biny wpadają w czasie
    # czyli zamienia wartosci wylosowane w czasie x(t) na biny w czasie
    bins_in_time = list()

    # zliczenia w binach
    # wartosc minimalna jest przypisywana rowniez do pierwszego binu
    epsilon = 1e-6
    for value in lst:
        if (value == xmin):
            bins_in_time.append(bin_center[0])
            counts[bin_center[0]] += 1
        elif (value == xmax):
            bins_in_time.append(bin_center[num_bin-1])
            counts[bin_center[num_bin-1]] += 1
        else:
            for k in range(num_bin):
                if (bin_center[k] - half_bin) < value <= (bin_center[k] + half_bin):
                    bins_in_time.append(bin_center[k])
                    counts[bin_center[k]] += 1

    # usuwa biny z za małą liczbą zliczeń
    for key in counts:
        if counts[key] < 100:
            bin_center.remove(key)

    # lista m z algorytmu dla odpowienich srodkow binu
    # implementacja algorytmu podesłanego. Oblicza M dla kolejnych środków binu
    m1 = []
    m2 = []

    # poniżej 'i' i 'k' są wartościami środka binu
    for i in bin_center:
        value1 = 0
        value2 = 0
        for k in bin_center[::-1]: # srodki binów, ale od konca
            if i != k:
                a = i_before_k(i, k, bins_in_time)
                b = occurrences_of_i(i, bins_in_time)
                if b != 0:
                    value1 += (k - i) *  a / b
                    value2 += (k - i)**2 *  a / b
        m1.append(value1)
        m2.append(value2)

    #Dopasowanie do danych D1
    #wielomian stopnia 2
    D1_poly_2 = np.polyfit(bin_center, calculate_D(m1, step, 1), 2)
    D1_poly_2_str = ['{:.1f}'.format(x) for x in D1_poly_2]
    D1_2_legend = '+'.join([f'{factor}x^{i-1}' for factor, i in zip(D1_poly_2_str, range(len(D1_poly_2_str), -1, -1))])
    # wielomian stopnia 3
    D1_poly_3 = np.polyfit(bin_center, calculate_D(m1, step, 1), 3)
    D1_poly_3_str = ['{:.1f}'.format(x) for x in D1_poly_3]
    D1_3_legend = '+'.join([f'{factor}x^{i-1}' for factor, i in zip(D1_poly_3_str, range(len(D1_poly_3_str), -1, -1))])
    # wielomian stopnia 4
    D1_poly_4 = np.polyfit(bin_center, calculate_D(m1, step, 1), 4)
    D1_poly_4_str = ['{:.1f}'.format(x) for x in D1_poly_4]
    D1_4_legend = '+'.join([f'{factor}x^{i-1}' for factor, i in zip(D1_poly_4_str, range(len(D1_poly_4_str), -1, -1))])

    # Dopasowanie wielomianami do D2
    # wielomian stopnia 2
    D2_poly_2 = np.polyfit(bin_center, calculate_D2(m2, step, 1), 2)
    D2_poly_2_str = ['{:.1f}'.format(x) for x in D2_poly_2]
    D2_2_legend = '+'.join([f'{factor}x^{i-1}' for factor, i in zip(D2_poly_2_str, range(len(D2_poly_2_str), -1, -1))])

    D2_poly_3 = np.polyfit(bin_center, calculate_D2(m2, step, 1), 3)
    D2_poly_3_str = ['{:.1f}'.format(x) for x in D2_poly_3]
    D2_3_legend = '+'.join([f'{factor}x^{i-1}' for factor, i in zip(D2_poly_3_str, range(len(D2_poly_3_str), -1, -1))])
    
        
    x = np.linspace(-3, 3)
 
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    ax1.scatter(bin_center, calculate_D(m1, step, 1), color='red', label='D1')
    ax1.plot(x, plot_poly(x, D1_poly_2, 2), label=D1_2_legend)
    ax1.plot(x, plot_poly(x, D1_poly_3, 3), label=D1_3_legend)
    ax1.plot(x, plot_poly(x, D1_poly_4, 4), label=D1_4_legend)
    ax1.plot(x, f1(x), label='x-x^3')
    ax2.scatter(bin_center, calculate_D2(m2, step, 2), color='red', label='D2')
    ax2.plot(x, plot_poly(x, D2_poly_2, 2), label=D2_2_legend)
    ax2.plot(x, plot_poly(x, D2_poly_3, 3), label=D2_3_legend)
    ax2.plot(x, f2(x), label='x^2+1 ')
    ax1.legend()
    ax1.grid()
    ax2.legend()
    ax2.grid()
    plt.show()

if __name__ == '__main__':
    co_to_jest()

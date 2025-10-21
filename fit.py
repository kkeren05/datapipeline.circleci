import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def fit_and_plot(filename='data.csv', plotfile='plot.png'):
    df = pd.read_csv(filename)
    x = df['X']
    y = df['Y']
    
    result = linregress(x, y)
    slope, intercept = result.slope, result.intercept

    plt.scatter(x, y, label='Data')
    plt.plot(x, slope * x + intercept, color='red', label='Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(plotfile)
    plt.close()
    print(f"Plot saved to {plotfile}")

    return slope, intercept

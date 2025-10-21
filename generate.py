import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# === STEP 1: Generate data ===
def generate_data():
    m = 2.0  # true slope
    b = 1.0  # true intercept
    x = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 1, size=100)
    y = m * x + b + noise

    data = pd.DataFrame({'X': x, 'Y': y})
    data.to_csv('data.csv', index=False)

    print("Data generated and saved to 'data.csv'")
    return m, b

# === STEP 2: Fit the data and plot ===
def fit_and_plot():
    data = pd.read_csv('data.csv')
    x = data['X']
    y = data['Y']

    # Linear regression
    result = linregress(x, y)
    slope = result.slope
    intercept = result.intercept

    # Plotting
    plt.scatter(x, y, label='Data', color='blue')
    plt.plot(x, slope * x + intercept, label='Fitted Line', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Fit to Noisy Data')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

    print("Plot saved to 'plot.png'")
    return slope, intercept

# === STEP 3: Simple Tests ===
def run_tests(true_m, true_b, fit_m, fit_b):
    print("\nRunning tests...\n")

    # Test 1: CSV file exists
    assert os.path.exists('data.csv'), "Test failed: data.csv does not exist"
    print("✔ Test passed: data.csv exists")

    # Test 2: Plot file exists
    assert os.path.exists('plot.png'), "Test failed: plot.png does not exist"
    print("✔ Test passed: plot.png exists")

    # Test 3: Check numeric values
    data = pd.read_csv('data.csv')
    assert data['X'].dtype.kind in 'fi', "Test failed: X values are not numeric"
    assert data['Y'].dtype.kind in 'fi', "Test failed: Y values are not numeric"
    print("✔ Test passed: Data values are numeric")

    # Test 4: Check if fitted values are close
    assert abs(true_m - fit_m) < 0.5, f"Test failed: slope is off (expected {true_m}, got {fit_m})"
    assert abs(true_b - fit_b) < 0.5, f"Test failed: intercept is off (expected {true_b}, got {fit_b})"
    print("✔ Test passed: Fitted line is close to original")

    print("\n✅ All tests passed!")


# === Main: Run the whole workflow ===
if __name__ == "__main__":
    m_true, b_true = generate_data()
    m_fit, b_fit = fit_and_plot()
    run_tests(m_true, b_true, m_fit, b_fit)

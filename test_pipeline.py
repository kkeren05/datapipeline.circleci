import os
import pandas as pd
import numpy as np
import pytest
from generate import generate_data
from fit import fit_and_plot

def test_csv_file_exists():
    generate_data()
    assert os.path.exists('data.csv')

def test_plot_file_exists():
    generate_data()
    fit_and_plot()
    assert os.path.exists('plot.png')

def test_data_numeric():
    generate_data()
    df = pd.read_csv('data.csv')
    assert np.issubdtype(df['X'].dtype, np.number)
    assert np.issubdtype(df['Y'].dtype, np.number)

def test_fit_accuracy():
    true_m, true_b = generate_data()
    fit_m, fit_b = fit_and_plot()
    tol = 0.5
    assert abs(true_m - fit_m) < tol
    assert abs(true_b - fit_b) < tol

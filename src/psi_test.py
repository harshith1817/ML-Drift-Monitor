import pandas as pd
import numpy as np

def calculate_psi(expected, actual, bins=10):
    expected=np.array(expected)
    actual=np.array(actual)
    
    breakpoints=np.percentile(expected, np.linspace(0, 100, bins+1))
    
    expected_counts=np.histogram(expected, bins=breakpoints)[0]
    actual_counts=np.histogram(actual, bins=breakpoints)[0]
    
    expected_perc=expected_counts/len(expected)
    actual_perc=actual_counts/len(actual)
    
    expected_perc=np.where(expected_perc==0, 0.0001, expected_perc)
    actual_perc=np.where(actual_perc==0, 0.0001, actual_perc)
    
    psi_values=(actual_perc-expected_perc)*np.log(actual_perc/expected_perc)
    
    psi=np.sum(psi_values)
    
    return float(psi)
import numpy as np
from web.categorization import calculate_purity

def test_purity():
    y_true = np.array([1,1,2,2,3])
    y_pred = np.array([2,2,2,2,1])
    assert abs(0.6 - calculate_purity(y_true, y_pred)) < 1e-10

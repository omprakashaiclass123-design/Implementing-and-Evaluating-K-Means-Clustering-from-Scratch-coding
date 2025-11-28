import numpy as np

def generate_synthetic_data(n_samples=600, random_state=42):
    np.random.seed(random_state)
    third = n_samples // 3
    c1 = np.random.randn(third, 2) + np.array([0, 0])
    c2 = np.random.randn(third, 2) + np.array([5, 5])
    c3 = np.random.randn(n_samples - 2*third, 2) + np.array([0, 5])
    return np.vstack([c1, c2, c3])

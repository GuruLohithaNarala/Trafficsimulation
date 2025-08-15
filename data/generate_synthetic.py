import pandas as pd
import numpy as np

def generate_synthetic_traffic_data(filename='data/synthetic_traffic.csv'):
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq='H')
    data = {
        'timestamp': timestamps,
        'node_1': np.random.poisson(lam=100, size=len(timestamps)),
        'node_2': np.random.poisson(lam=80, size=len(timestamps)),
        'node_3': np.random.poisson(lam=90, size=len(timestamps)),
        'node_4': np.random.poisson(lam=95, size=len(timestamps)),
        'node_5': np.random.poisson(lam=85, size=len(timestamps)),
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    generate_synthetic_traffic_data()

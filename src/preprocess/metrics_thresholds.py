import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "construct_data"

def load_metric_data(metric_dir):
    """
    Load and combine metric data from multiple files
    
    Args:
        metric_dir: Directory containing metric files
        
    Returns:
        DataFrame: Combined metrics data
    """
    metric_files = [f for f in os.listdir(metric_dir) if 'metric' in f]
    metrics_data = [
        pd.read_csv(
            os.path.join(metric_dir, f), 
            usecols=['CpuUsageRate(%)', 'MemoryUsageRate(%)']
        ) 
        for f in metric_files
    ]
    return pd.concat(metrics_data, ignore_index=True)

def train_anomaly_detector(X_scaled):
    """
    Train isolation forest model for anomaly detection
    
    Args:
        X_scaled: Scaled input data
        
    Returns:
        tuple: (model, labels)
    """
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_scaled)
    return model, model.predict(X_scaled)

def calculate_resource_thresholds(X, labels):
    """
    Calculate resource thresholds from model predictions
    
    Args:
        X: Raw input data
        labels: Model predictions
        
    Returns:
        dict: Threshold values for CPU and memory
    """
    normal_indices = labels == 1 if 1 in labels else labels == 0
    return {
        'CPU Threshold': np.percentile(X[normal_indices, 0], 99),
        'Memory Threshold': np.percentile(X[normal_indices, 1], 99)
    }

def calculate_global_resource_thresholds(metric_dir):
    """
    Calculate global resource thresholds using anomaly detection
    
    Args:
        metric_dir: Directory containing metric files
        
    Returns:
        dict: Model results with thresholds
    """
    # Load and prepare data
    combined_metrics = load_metric_data(metric_dir)
    X = combined_metrics.values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model and get predictions
    model, labels = train_anomaly_detector(X_scaled)
    
    # Calculate thresholds
    thresholds = calculate_resource_thresholds(X, labels)
    
    return {'Isolation Forest': thresholds}

def main():
    """Calculate metric thresholds and save"""
    metric_dir = TRAIN_DATA_DIR /'2022-08-22/metric'
    thresholds = calculate_global_resource_thresholds(metric_dir)
    
    cpu = thresholds['Isolation Forest']['CPU Threshold']
    mem = thresholds['Isolation Forest']['Memory Threshold']
    result = round((cpu + mem) / 2)
    
    output = {
        'global_threshold': result,
        'cpu_threshold': cpu,
        'memory_threshold': mem
    }
    
    print(f"Metrics thresholds: {output}")
    print(f"Global threshold: {result}")

if __name__ == '__main__':
    main()

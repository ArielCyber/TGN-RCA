import os
import json

# Convert path to use forward slashes
ROOT_DIR = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/')

# Datasets
ONLINE_BOUTIQUE = "OnlineBoutique"
TRAIN_TICKET = "TrainTicket"

# Directories
DATASET_DIR = f"{ROOT_DIR}/dataset"
RCA_DATA_DIR = f"{DATASET_DIR}/rca_data"
TRAIN_DATA_DIR = f"{DATASET_DIR}/construct_data"
METRIC_THRESHOLD_DATA_DIR = f"{DATASET_DIR}/metric_threshold"
LOG_TEMPLATE_DIR = f"{ROOT_DIR}/src/preprocess/log_template"
ANALYSIS_DIR = f"{ROOT_DIR}/analysis"

LOG_DIR = f"{ROOT_DIR}/log"
MODEL_DIR = f"{ROOT_DIR}/model"

class ThresholdConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            with open(f"{METRIC_THRESHOLD_DATA_DIR}/thresholds.json", 'r') as f:
                cls._instance.thresholds = json.load(f)
        return cls._instance.thresholds

METRICS_THRESHOLDS = ThresholdConfig()

# Dataset Configurations
DATASET_CONFIGS = {
    ONLINE_BOUTIQUE: {
        "normal_times": ["2022-08-22 03:51", "2022-08-23 17:00"],
        "paths": [
            f"{DATASET_DIR}/rca_data/2022-08-22/2022-08-22-fault_list.json",
            f"{DATASET_DIR}/rca_data/2022-08-23/2022-08-23-fault_list.json"
        ]
    },
    TRAIN_TICKET: {
        "normal_times": ["2023-01-29 08:50", "2023-01-30 11:39"],
        "paths": [
            f"{DATASET_DIR}/rca_data/2023-01-29/2023-01-29-fault_list.json",
            f"{DATASET_DIR}/rca_data/2023-01-30/2023-01-30-fault_list.json"
        ]
    }
}

DEFAULT_EPOCHS = 50
DEFAULT_NORMAL_SCORE = 1.0
DEFAULT_ABNORMAL_SCORE = 0.9999
MIN_OCCURANCES = 5

with open(f"{METRIC_THRESHOLD_DATA_DIR}/thresholds.json", 'r') as f:
        THRESHOLDS = json.load(f)
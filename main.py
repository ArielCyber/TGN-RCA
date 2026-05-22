import os
from dotenv import load_dotenv
from src.preprocess.log_parser import get_miner
from src.record_sequences.sequence_ranker import rca
from src.log.log_handler import get_logger
from src.config import *

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

def run_evaluation(dataset, config, train_data_path):
    """Execute evaluation for given dataset"""
    log_template = get_miner(dataset)
    logger.info(f"Starting RCA for dataset: {dataset}")
    rca(config["normal_times"], config["paths"], train_data_path, dataset, log_template)

def setup_directories(dataset):
    """Create and clean analysis and model directories"""
    dataset_analysis_dir = f"{ANALYSIS_DIR}/{dataset}"
    dataset_model_dir = f"{MODEL_DIR}/{dataset}"
    
    # Create directories
    os.makedirs(dataset_analysis_dir, exist_ok=True)
    os.makedirs(dataset_model_dir, exist_ok=True)
    
    # Clean existing files
    for file in os.listdir(dataset_analysis_dir):
        os.remove(os.path.join(dataset_analysis_dir, file))
    for file in os.listdir(dataset_model_dir):
        os.remove(os.path.join(dataset_model_dir, file))


def main():
    # Get values from .env
    dataset = os.getenv('DATASET', "")
    setup_directories(dataset)
    run_evaluation(dataset, DATASET_CONFIGS[dataset], TRAIN_DATA_DIR)

if __name__ == '__main__':
    main()

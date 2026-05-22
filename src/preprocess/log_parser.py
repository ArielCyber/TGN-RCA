from email import message
import os
import re
import json
import pandas as pd
from os.path import dirname
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from src.log.log_handler import get_logger
from src.config import *

logger = get_logger(__name__)

def get_miner(dataset):
    config = TemplateMinerConfig()
    config.load(f"{LOG_TEMPLATE_DIR}/drain3_{dataset}.ini")
    config.profiling_enabled = False

    path = f"{LOG_TEMPLATE_DIR}/{dataset}.bin"
    persistence = FilePersistence(path)
    template_miner = TemplateMiner(persistence, config=config)

    return template_miner

def save_logs_to_csv(cluster_id, log_message, template):
    """
    Save log parsing results to CSV file, only if ID doesn't exist
    """
    csv_file = f"{ANALYSIS_DIR}/{os.getenv('DATASET')}/logs.csv"
    
    # Read existing CSV if it exists
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        if cluster_id in existing_df['id'].values:
            return
    
    # Create new entry
    df = pd.DataFrame({
        'id': [cluster_id],
        'log_message': [log_message],
        'template': [template]
    })
    
    # Append to file
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)




def log_blueprint_parse(log, pod, log_template, logrca=False):
    """
    Parse logs using Drain3 template mining algorithm.
    
    Args:
        log (str): Raw log message to be parsed
        pod (str): Name of the pod generating the log
        log_template: Drain3 template miner instance
        logrca (bool): Flag to append pod name to log message
        
    Returns:
        int: Cluster ID assigned to the parsed log message
        
    Notes:
        Output contains:
        - change_type: Template identification status (new/changed/existing)
        - cluster_id: Sequential cluster identifier
        - cluster_size: Number of messages in cluster
        - cluster_count: Total clusters processed
        - template_mined: Latest template pattern
    """
    # Extract service information from log
    log_message, service = extract_service_name(log, pod)
    
    # Append pod name for LogRCA if enabled
    if logrca:
        log_message = f"{log_message}_{pod}"
    
    # Process log through template miner
    result = log_template.add_log_message(log_message)
    
    # Log template changes
    if result["change_type"] != "none":
        result_json = json.dumps(result)
        logger.info(f"{service} Log Parsing Result: {result_json}")
    
    # Persist results to CSV
    save_logs_to_csv(
        cluster_id=result['cluster_id'],
        log_message=log_message,
        template=result['template_mined']
    )
    
    return result['cluster_id']


def id_to_blueprint(template_id, log_template):
    """
    Retrieves the template content for a given template ID.
    
    Args:
        template_id: Unique identifier of the template
        log_template: Log template miner object containing clusters
        
    Returns:
        str: The template content if found, empty string otherwise
    """
    return next(
        (cluster.get_template() for cluster in log_template.drain.clusters 
         if cluster.cluster_id == template_id),
        ""
    )


def extract_service_name(log, pod):
    """
    Extract service name from pod and parse log message based on service type.
    
    Args:
        log (str): Raw log entry
        pod (str): Pod name
        
    Returns:
        tuple: (log_message, service_name)
    """
    service_patterns = {
        'adservice': lambda l: json.loads(l)['log'],
        'cartservice': lambda l: json.loads(l)['log'],
        'checkoutservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'currencyservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'emailservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'frontend': lambda l: json.loads(json.loads(l)['log'])['message'],
        'paymentservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'productcatalogservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'recommendationservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'shippingservice': lambda l: json.loads(json.loads(l)['log'])['message'],
        'alarm': lambda l: l
    }

    try:
        # Handle ts- services separately
        if pod.startswith('ts-'):
            service = pod.rsplit('-', 2)[0]
            log_message = json.loads(log)['log']
            
            # Extract message using regex patterns
            patterns = [r"  (.+?#.+?) ", r" (.+?#.+?) "]
            for pattern in patterns:
                matches = re.findall(pattern, log_message)
                if matches:
                    return matches[0].rstrip(), service
            
            logger.error(f"Regex failed for log message: {log_message}")
            return log_message.rstrip(), service

        # Handle other services
        for service_name, parser in service_patterns.items():
            if service_name in pod:
                return parser(log).rstrip(), service_name

        logger.fatal(f"Unknown pod: {pod}")
        return log.rstrip(), ""

    except Exception:
        return log.rstrip(), ""


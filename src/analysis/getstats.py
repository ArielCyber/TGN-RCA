import pandas as pd
import glob
import os

def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipping file {file_path} due to: {str(e)}")
        return pd.DataFrame()


def count_all_data(base_path, system_dates):
    total_metrics = 0
    total_traces = 0
    total_spans = 0
    total_logs = 0
    
    for date in system_dates:
        # Count metrics
        metric_files = glob.glob(os.path.join(base_path, date, "metric/*_metric.csv"))
        for file in metric_files:
            df = safe_read_csv(file)
            total_metrics += len(df)
            
        # Count traces (from traceid files)
        traceid_files = glob.glob(os.path.join(base_path, date, "traceid/*.csv"))
        for file in traceid_files:
            df = safe_read_csv(file)
            total_traces += len(df)
            
        # Count spans (from trace files)
        trace_files = glob.glob(os.path.join(base_path, date, "trace/*.csv"))
        for file in trace_files:
            df = safe_read_csv(file)
            total_spans += len(df)
            
        # Count logs
        log_files = glob.glob(os.path.join(base_path, date, "log/*.csv"))
        for file in log_files:
            df = safe_read_csv(file)
            total_logs += len(df)
    
    return total_metrics, total_traces, total_spans, total_logs

#base_path = "./dataset/construct_data"
base_path = "./dataset/rca_data"
online_boutique_dates = ["2022-08-22", "2022-08-23"]

# Get counts for both systems
ob_metrics, ob_traces, ob_spans, ob_logs = count_all_data(base_path, online_boutique_dates)

print("OnlineBoutique:")
print(f"Metrics: {ob_metrics}")
print(f"Traces: {ob_traces}")
print(f"Spans: {ob_spans}")
print(f"Logs: {ob_logs}")

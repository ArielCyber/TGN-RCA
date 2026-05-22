from sklearn.ensemble import IsolationForest
from src.preprocess.data_processor import *
from src.record_sequences import *
from src.preprocess import *
from src.tgn import *
from collections import deque
from src.config import *
from src.log.log_handler import get_logger
logger = get_logger(__name__)

def compute_span_depth(span, span_dict, span_depths, cache=None):
    """
    Calculate the depth of a span in the trace hierarchy
    
    Args:
        span: Span object to compute depth for
        span_dict: Dictionary mapping span IDs to spans
        span_depths: Dictionary to store computed depths
        cache: Optional cache dictionary for memoization
        
    Returns:
        int: Computed depth of the span
    """
    if cache is None:
        cache = {}
        
    # Return cached result if available
    if span.spanid in cache:
        return cache[span.spanid]
        
    # Calculate depth based on parent relationship
    if span.parentid == 'root' or span.parentid not in span_dict:
        depth = 0
    else:
        parent_span = span_dict.get(span.parentid)
        depth = 1 + compute_span_depth(parent_span, span_dict, span_depths, cache) if parent_span else 0
    
    # Store results
    cache[span.spanid] = depth
    span_depths[span.spanid] = depth
    
    return depth


def sequences_score_old(record_sequences):
    """
    Calculate sequence values across multiple records
    
    Args:
        record_sequences: List of record sequence objects
        
    Returns:
        dict: Sorted sequence dictionary
    """
    sequence_dict = {}
    total_sequences = set()
   
    # Aggregate values across sequences
    for sequences in record_sequences:
        for sequence, value in sequences.sequence_counts.items():
            if sequence in total_sequences:
                sequence_dict[sequence] += value
            else:
                sequence_dict[sequence] = value
        total_sequences.update(sequences.unique_pairs)
   
    # Sort by value in descending order
    return dict(sorted(sequence_dict.items(), key=lambda x: x[1], reverse=True))

def sequences_score(record_graphs):
    """
    Calculate sequence values across multiple records
    
    Args:
        record_graphs: List of record graph objects
        
    Returns:
        dict: Sorted sequence dictionary
    """
    sequence_dict = {}
    total_pairs = set()
    
    # Aggregate values across sequences
    for graph in record_graphs:
        # Extract sequences from each graph
        graph.extract_sequences()
        #graph.show_sequences()
        #logger.debug(f"Graph sequences items: {graph.sequence_counts.items()}")
        for key, value in graph.sequence_counts.items():
            if key in total_pairs:
                sequence_dict[key] += value
            else:
                sequence_dict[key] = value
        total_pairs.update(graph.unique_pairs)
    
    # Sort by value in descending order
    return dict(sorted(sequence_dict.items(), key=lambda x: x[1], reverse=True))

def sequences_tgn_score(edge_list, link_embeddings, mode):
    """
    Calculate sequence scores using vectorized operations
   
    Args:
        edge_list: List of (source, destination) edge tuples
        link_embeddings: Dictionary of edge embeddings
        mode: Analysis mode ('train' or 'inference')
       
    Returns:
        dict: Sorted dictionary of sequence scores
    """
    #logger.debug(f"Processing {len(edge_list)} edges for sequence scoring")
   
    # Prepare arrays for vectorized operations
    sequence_keys = [f"{src}_{dst}" for src, dst in edge_list]
    scores = np.zeros(len(edge_list), dtype=np.float32)
   
    # Calculate scores
    for i, (src, dst) in enumerate(edge_list):
        embedding = link_embeddings.get(f"{src}_{dst}", [0.0])
        scores[i] = embedding[-1] if mode == 'inference' else np.mean(embedding)
   
    # Sort results efficiently
    sorted_indices = np.argsort(-scores)
    sorted_sequences = [sequence_keys[i] for i in sorted_indices]
    sorted_scores = scores[sorted_indices]
   
    return dict(zip(sorted_sequences, sorted_scores))

def get_sequences_gnn(alarm_list, metric_list, log_template, sequence_scores, 
                     mode, record_sequences, log_sequences, log_metrics, train_epoch, timestamp):
    """
    Extract GNN-based sequence sequences from logs, traces, and metrics
    
    Args:
        alarm_list: List of system alarms
        metric_list: List of system metrics
        log_template: Template mining instance
        sequence_scores: Dictionary of sequence scores
        mode: Analysis mode ('train' or 'inference')
        record_sequences: List of record sequences
        log_sequences: List of log sequences
        log_metrics: Dictionary of log-based metrics
        train_epoch: Number of training epochs
    
    Returns:
        tuple: (sequence_scores_tgn, metrics_dict)
    """
    # Build a mapping from pods to their alarms at specific times
    alarm_metrics = {}
    for alarm in alarm_list:
        pod = alarm['pod']
        if pod not in alarm_metrics:
            alarm_metrics[pod] = len(alarm['alarm'])
        else:
            alarm_metrics[pod] += len(alarm['alarm'])

    # Build a dictionary mapping each pod to its metrics
    pod_metrics = {}
    for pod_metric in metric_list:
        pod = pod_metric['pod']
        metrics = pod_metric['metrics']
        metric_dict = {metric['metric_type']: metric['metric_value'] for metric in metrics}
        pod_metrics[pod] = metric_dict
   
    # Build span mappings
    span_dict, span_depths = _build_span_mappings(log_sequences)
   
    # Build record mappings
    record_trace_ids = _build_record_mappings(log_sequences)
   
    # Generate node and edge data
    nodes_df = _generate_node_data(log_sequences, pod_metrics, alarm_metrics,
                                 span_depths, log_metrics)
    edges_df = _generate_edge_data(record_sequences, record_trace_ids, sequence_scores)
   
    # Train or infer using TGN
    if mode == 'train':
        logger.info("Training TGN model")
        tgn_train(nodes_df, edges_df, train_epoch, timestamp)
   
    # Get TGN results
    _, link_embeddings, edge_list, metrics_dict = tgn_inference(nodes_df, edges_df, mode, timestamp)
    sequence_scores_tgn = sequences_tgn_score(edge_list, link_embeddings, mode)
   
    return sequence_scores_tgn, metrics_dict


def _process_alarms(alarm_list):
    """Process alarms and build alarm metrics dictionary"""
    alarm_metrics = {}
    for alarm in alarm_list:
        pod = alarm['pod']
        alarm_metrics[pod] = alarm_metrics.get(pod, 0) + len(alarm['alarm'])
    return alarm_metrics

def _process_metrics(metric_list):
    """Process metrics and build pod metrics dictionary"""
    return {
        pod_metric['pod']: {
            metric['metric_type']: metric['metric_value'] 
            for metric in pod_metric['metrics']
        }
        for pod_metric in metric_list
    }

def _build_span_mappings(log_sequences):
    """Build span dictionary and compute depths"""
    span_dict = {
        span.spanid: span
        for trace in log_sequences
        for span in trace.spans
    }
    
    span_depths = {}
    for span in span_dict.values():
        compute_span_depth(span, span_dict, span_depths)
    
    return span_dict, span_depths

def _build_record_mappings(log_sequences):
    """Build record to trace ID mappings"""
    return {
        (record.record, record.timestamp): trace.traceid
        for trace in log_sequences
        for span in trace.spans
        for record in span.records
    }


def save_mapping_to_csv(mapping, dataset):
    """
    Save record mappings to CSV file with timestamp, record_id, traceid and spanid
    """
    csv_file = f"{ANALYSIS_DIR}/{dataset}/mapping.csv"
    df = pd.DataFrame(mapping)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)



def _generate_node_data(log_sequences, pod_metrics, alarm_metrics, span_depths, log_metrics):
    """Generate node DataFrame"""
    nodes_data = []
    mapping = []
    nodes_set = set()
   
    for trace in log_sequences:
        record_counts = _compute_record_counts(trace)
        total_records = sum(record_counts.values())
       
        nodes_data.extend(_create_node_entries(
            trace, record_counts, total_records,
            pod_metrics, alarm_metrics, span_depths,
            log_metrics, nodes_set
        ))

        mapping.extend(_create_node_mapping(trace))


    # Save mapping to CSV
    save_mapping_to_csv(mapping, os.getenv('DATASET'))

    return pd.DataFrame(nodes_data)

def _generate_edge_data(record_sequences, record_trace_ids, sequence_scores):
    """Generate edge DataFrame"""
    edges_data = []
   
    for sequence in record_sequences:
        for source_record in sequence.adjacency_dict:
            for target_record in sequence.adjacency_dict[source_record]:
                edge_data = _create_edge_entry(
                    source_record, target_record,
                    record_trace_ids, sequence_scores
                )
                edges_data.append(edge_data)
   
    return pd.DataFrame(edges_data)

def _compute_record_counts(trace):
    """
    Calculate record occurrence counts in a trace
   
    Args:
        trace: Trace object containing spans and records
       
    Returns:
        dict: Mapping of record IDs to their counts
    """
    record_counts = {}
    for span in trace.spans:
        for record in span.records:
            record_counts[record.record] = record_counts.get(record.record, 0) + 1
    return record_counts


def _create_node_mapping(trace):
    mapping = []

    for span in trace.spans:
        for record in span.records:
            mapping.append({
                'record_id': record.record,
                'timestamp': record.timestamp,
                'traceid': trace.traceid,
                'spanid': span.spanid
            })


    return mapping


def _create_node_entries(trace, record_counts, total_records, pod_metrics,
                        alarm_metrics, span_depths, log_metrics, nodes_set):
    """
    Create node entries for each record in a trace
   
    Args:
        trace: Trace object
        record_counts: Dictionary of record counts
        total_records: Total number of records
        pod_metrics: Dictionary of pod metrics
        alarm_metrics: Dictionary of alarm counts
        span_depths: Dictionary of span depths
        log_metrics: Dictionary of log metrics
        nodes_set: Set of processed nodes
       
    Returns:
        list: Node data entries
    """
    nodes = []


   
    for span in trace.spans:
        for record in span.records:
            record_key = (record.record, record.timestamp)
            if record_key in nodes_set:
                continue
               
            nodes_set.add(record_key)
            metrics = pod_metrics.get(record.pod, {})
           
            nodes.append({
                'record_id': record.record,
                'timestamp': record.timestamp,
                'MemoryUsageRate(%)': metrics.get('MemoryUsageRate(%)', np.nan),
                'CpuUsageRate(%)': metrics.get('CpuUsageRate(%)', np.nan),
                'NetworkP90Latency(ms)': metrics.get('NetworkP90(ms)', np.nan),
                'AlarmCount': alarm_metrics.get(record.pod, 0),
                'IsRoot': int(span.parentid == 'root'),
                'Depth': span_depths.get(span.spanid, 0),
                'RecordRatio': record_counts[record.record] / total_records if total_records > 0 else 0.0,
                'ErrorRatio': log_metrics.get(record.pod, 0)
            })


    return nodes


def _create_edge_entry(source_record, target_record, record_trace_ids, sequence_scores):
    """
    Create edge entry for a record pair
   
    Args:
        source_record: Source record object
        target_record: Target record object
        record_trace_ids: Dictionary mapping records to trace IDs
        sequence_scores: Dictionary of sequence scores
       
    Returns:
        dict: Edge data entry
    """
    source_key = (source_record.record, source_record.timestamp)
    sequence_key = f"{source_record.record}_{target_record.record}"
   
    return {
        'source_record_id': source_record.record,
        'target_record_id': target_record.record,
        'timestamp': source_record.timestamp,
        'weight': sequence_scores[sequence_key],
        'TraceID': record_trace_ids.get(source_key)
    }


def save_alarms_to_csv(alarms, timestamp, mode, dataset):
    """
    Save alarms to CSV file with mode, pod, and metric columns
    """
    csv_file = f"{ANALYSIS_DIR}/{dataset}/alarms.csv"
    rows = []
    
    for alarm_entry in alarms:
        pod = alarm_entry['pod']
        for alarm in alarm_entry['alarm']:
            rows.append({
                'mode' :mode,
                'timestamp': timestamp,
                'pod': pod,
                'metric': alarm['metric_type']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def save_sequences_to_csv(sequence_dict, timestamp, mode, dataset):
    """
    Save sequence scores to CSV file
    """
    csv_file = f"{ANALYSIS_DIR}/{dataset}/sequences.csv"
    rows = []
    
    for key, count in sequence_dict.items():
        source, target = key.split('_')
        rows.append({
            'mode': mode,
            'timestamp': timestamp,
            'source': source,
            'target': target,
            'count': count
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)


def get_sequence(timestamp, dataset, data_path, template_miner, mode, train_epoch=DEFAULT_EPOCHS):
    """
    Extract record sequences from system data at a specific timestamp
   
    Args:
        timestamp: Target timestamp for sequence extraction
        dataset: Name of the dataset
        data_path: Base directory for data files
        template_miner: Log template mining instance
        mode: Analysis mode ('train' or 'inference')
        train_epoch: Number of training epochs
       
    Returns:
        tuple: (support_list, record_sequences, alarms, gnn_support_list, gnn_metrics)
    """
    # Get file paths
    file_paths = _get_data_file_paths(timestamp, data_path)
    logger.info(f"file_paths: {file_paths}")
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"data_path: {data_path}")
    # Get metrics and alarms
    metrics, log_metrics = get_metrics(timestamp, data_path)
    alarms = set_raised_metric(metrics, dataset, data_path)

    save_alarms_to_csv(alarms, timestamp, mode, dataset)
    
    # Log data collection results
    _log_data_collection(metrics, log_metrics, alarms)
    
    # Integrate data sources
    record_sequences, log_sequences = integrate_multimodal_data(
        file_paths['trace'],
        file_paths['trace_id'],
        file_paths['log'],
        alarms,
        dataset,
        template_miner
    )
   
    #logger.info(f"record_sequences: {record_sequences}")
    # Calculate sequence support
    sequence_dict = sequences_score(record_sequences)

    save_sequences_to_csv(sequence_dict, timestamp, mode, dataset)

    logger.info(f"Sequence support: {sequence_dict}")
   
    # Get GNN-based sequences
    gnn_support, gnn_metrics = get_sequences_gnn(
        alarms, metrics, template_miner, sequence_dict,
        mode, record_sequences, log_sequences, log_metrics, train_epoch, timestamp
    )
   
    return sequence_dict, record_sequences, alarms, gnn_support, gnn_metrics


def _get_data_file_paths(timestamp, base_path):
    """
    Generate file paths for data sources
    
    Args:
        timestamp: Target timestamp in format "YYYY-MM-DD HH:MM"
        base_path: Base directory path
        
    Returns:
        dict: Paths for trace, trace_id and log files
    """
    date, time = timestamp.split(" ")
    hour, minute = time.split(":")[:2]
    
    paths = {
        'trace': f"{base_path}/{date}/trace/{hour}_{minute}_trace.csv",
        'trace_id': f"{base_path}/{date}/traceid/{hour}_{minute}_traceid.csv", 
        'log': f"{base_path}/{date}/log/{hour}_{minute}_log.csv"
    }
    
    return paths

def _log_data_collection(metrics, log_metrics, alarms):
    """
    Log collected data metrics and alarms
    
    Args:
        metrics: System metrics data
        log_metrics: Log-based metrics
        alarms: Generated alarms
    """
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Log-based metrics: {log_metrics}") 
    logger.info(f"Alarms: {alarms}")

 

def calculate_dependency_strength(record_sequences):
    dependency_counts = defaultdict(int)
    total_calls = defaultdict(int)
   
    for sequence in record_sequences:
        for source_record in sequence.adjacency_dict:
            source_id = source_record.record
            targets = sequence.adjacency_dict[source_record]
            for target_record in targets:
                target_id = target_record.record
                key = f"{source_id}_{target_id}"
                dependency_counts[key] += 1
                total_calls[source_id] += 1
   
    dependency_weights = {}
    for key, count in dependency_counts.items():
        source_id = key.split('_')[0]
        dependency_weights[key] = count / total_calls[int(source_id)]
   
    return dependency_weights

def calculate_record_impact(record_sequences):
    record_pair_impact = {}
   
    for sequence in record_sequences:
        for source_record in sequence.adjacency_dict:
            source_id = source_record.record
            for target_record in sequence.adjacency_dict[source_record]:
                target_id = target_record.record
                key = f"{source_id}_{target_id}"
               
                downstream_records = set()
                queue = deque([target_record])
               
                while queue:
                    current = queue.popleft()
                    if current in sequence.adjacency_dict:
                        for next_record in sequence.adjacency_dict[current]:
                            next_id = next_record.record
                            if next_id not in downstream_records:
                                downstream_records.add(next_id)
                                queue.append(next_record)
               
                impact = len(downstream_records)
                if key not in record_pair_impact or impact > record_pair_impact[key]:
                    record_pair_impact[key] = impact
                   
    return record_pair_impact


def calculate_all_record_depths(record_sequences):
    record_depths = {}
    record_pods = {}
    record_timestamps = {}
   
    for sequence in record_sequences:
        for source_record in sequence.adjacency_dict:
            source_id = source_record.record
            depth, pod = sequence.calc_pod_depth(source_id)
            timestamp = source_record.timestamp
            classified_depth = depth
            
            if source_id not in record_depths or classified_depth > record_depths[source_id]:
                record_depths[source_id] = classified_depth
                record_pods[source_id] = pod
                record_timestamps[source_id] = timestamp
           
    return record_depths, record_pods, record_timestamps

def calculate_all_path_durations(record_sequences):
    path_durations = {}
   
    for sequence in record_sequences:
        for target_record in sequence.nodes:
            duration = 0
            current_record = target_record
            start_time = None
            end_time = None
           
            while True:
                found = False
                for key in sequence.adjacency_dict.keys():
                    for item in sequence.adjacency_dict[key]:
                        if current_record == item.record:
                            if end_time is None:
                                end_time = item.timestamp
                            current_record = key.record
                            start_time = key.timestamp
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
                   
            if start_time and end_time:
                duration = end_time - start_time
            path_durations[target_record] = duration
           
    return path_durations


def calculate_record_fanout(record_sequences):
    record_fanout = {}
    record_pods = {}
   
    for sequence in record_sequences:
        for source_record in sequence.adjacency_dict:
            source_id = source_record.record
            pod = source_record.pod
           
            # Count number of downstream records
            fanout = len(sequence.adjacency_dict[source_record])
           
            # Keep track of maximum fanout for each record
            if source_id not in record_fanout or fanout > record_fanout[source_id]:
                record_fanout[source_id] = fanout
                record_pods[source_id] = pod
               
    return record_fanout, record_pods

def calculate_final_scores(dependency_weights, record_depths, record_pods, record_timestamps, record_duration):
    max_depth = max(record_depths.values())
    final_scores = {}
   
    for record_pair, dep_strength in dependency_weights.items():
        source_id = int(record_pair.split('_')[0])
        depth = record_depths.get(source_id, 0)
        pod = record_pods.get(source_id, '')
        recordtime = record_timestamps.get(source_id, 0)
        dur = record_duration.get(source_id, 0)
       
        # Normalize depth to 0-1 range
        normalized_depth = depth / max_depth
       
        # Calculate combined score
        final_score = (0.8 * normalized_depth) + (0.2 * dep_strength)
       
        final_scores[record_pair] = {
            'score': final_score,
            'depth': depth,
            'pod': pod,
            'dependency_strength': dep_strength,
            'recordtime': recordtime,
            'duration': dur
        }
   
    return final_scores


def sequences_threshold(score_dict):
    # Convert scores to numpy array
    scores = np.array([score for score in score_dict.values() if score != 1.0]).reshape(-1, 1)
    
    # Scale the scores
    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.005, random_state=42)
    iso_forest.fit(scores_scaled)
    
    # Get predictions
    labels = iso_forest.predict(scores_scaled)
    
    # Calculate threshold from normal points (labeled as 1)
    threshold = np.percentile(scores[labels == 1], 99.99)
    
    return float(threshold)


def sequences_threshold_contamination(score_dict,contamination):
    # Convert scores to numpy array
    scores = np.array([score for score in score_dict.values() if score != 1.0]).reshape(-1, 1)
    
    
    # Scale the scores
    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(scores_scaled)
    
    # Get predictions
    labels = iso_forest.predict(scores_scaled)
    
    # Calculate threshold from normal points (labeled as 1)
    threshold = np.percentile(scores[labels == 1], 99.99)
    
    return float(threshold)


def analyze_sequences(normal_sequences, abnormal_time, dataset, template_miner,
                    normal_sequences_gnn, dependency_scores):
    """
    Analyze and rank sequences based on normal and abnormal temporal sequence score
    """
    abnormal_data = _get_abnormal_sequences(abnormal_time, dataset, template_miner)
    abnormal_sequences, alarm_list, abnormal_sequences_gnn, abnormal_metrics_gnn = abnormal_data
   
    score_data = _calculate_sequence_scores(
        normal_sequences, abnormal_sequences,
        normal_sequences_gnn, abnormal_sequences_gnn
    )
   
    _log_sequence_analysis(score_data, abnormal_metrics_gnn, alarm_list, template_miner)
    filtered_scores = _filter_sequences(score_data['scores'], score_data['thresholds'])
    logger.info(f"Filtered scores: {filtered_scores}")
    root_sequences = _filter_root_sequences(filtered_scores, template_miner)
    logger.info(f"Filtered root: {filtered_scores}")
    results = _generate_results(root_sequences, dependency_scores, alarm_list)
    processed_results = _process_alarms(results, alarm_list)
    logger.info(f"Result before sorting: {processed_results}")
    return sorted(processed_results, key=custom_sort_key)


def _get_abnormal_sequences(abnormal_time, dataset, template_miner):
    """Get sequences from abnormal period"""
    sequences = get_sequence(abnormal_time, dataset, RCA_DATA_DIR, 
                         template_miner, mode='inference')
    return (sequences[0], sequences[2], sequences[3], sequences[4])

def _calculate_sequence_scores(normal_sequences, abnormal_sequences, 
                            normal_sequences_gnn, abnormal_sequences_gnn):
    """Calculate scores for sequence comparison"""
    scores = {'scores': {}, 'thresholds': {}, 'normal': {}, 'abnormal': {}}
    logger.info(f"Normal sequences: {normal_sequences}")
    logger.info(f"Abnormal sequences: {abnormal_sequences}")
    logger.info(f"Normal sequences GNN: {normal_sequences_gnn}")
    logger.info(f"Abnormal sequences GNN: {abnormal_sequences_gnn}")
    for key in normal_sequences:
        if not _is_valid_sequence(key, normal_sequences):
            continue
            
        normal_score = _normalize_gnn_score(normal_sequences_gnn[key])
        
        if key in abnormal_sequences:
            abnormal_score = _normalize_gnn_score(abnormal_sequences_gnn[key])
            scores = _update_scores(scores, key, normal_score, abnormal_score)
        else:
            scores = _handle_missing_sequence(scores, key, normal_score)
            
    return scores

def _filter_sequences(scores, thresholds):
    """Filter sequences based on threshold"""
    threshold = sequences_threshold(thresholds)
    logger.info(f"Threshold: {threshold}")
    return {k:v for k,v in scores.items() if float(v) >= threshold}

def _filter_root_sequences(scores, template_miner):
    """Filter sequence based on root cause analysis"""
    filtered = {}
    for key, value in scores.items():
        if _is_valid_root_sequence(key, scores, template_miner):
            filtered[key] = value
    return filtered

def _generate_results(sequences, dependency_scores, alarm_list):
    """Generate result list with metadata"""
    results = []
    depth_dict = {}
    
    for key, value in sequences.items():
        result = _create_result_entry(
            key, value, dependency_scores[key], 
            depth_dict, alarm_list
        )
        if result:
            results.append(result)
            
    return results, depth_dict

def _process_alarms(results_data, alarm_list):
    """Process and filter alarm-related results"""
    results, depth_dict = results_data
    filtered_indices = _get_filtered_indices(results, alarm_list, depth_dict)
    return [r for idx, r in enumerate(results) if idx not in filtered_indices]

def _log_sequence_analysis(score_data, abnormal_metrics, alarms, template_miner):
    """Log detailed sequence analysis results"""
    logger.info(f"Sequence Scores: {score_data['scores']}")
    logger.info(f"Normal Sequence Data: {score_data['normal']}")
    logger.info(f"Abnormal Sequence Data: {score_data['abnormal']}")
    
    for key, value in score_data['scores'].items():
        src, dst = map(int, key.split('_'))
        depth = abnormal_metrics.get(key, [0]*6)[5]
        logger.info(
            f"Sequence {key}: src={src}, dst={dst}, score={value}, "
            f"template_src={id_to_blueprint(src, template_miner)}, "
            f"template_dst={id_to_blueprint(dst, template_miner)}, "
            f"depth={depth}, alarms={alarms}"
        )

def _is_valid_sequence(key, sequences):
    """Check if sequence meets minimum requirements"""
    return sequences[key] > MIN_OCCURANCES or unittest_val2(key)

def _normalize_gnn_score(score):
    """Normalize GNN scores using exponential"""
    return torch.exp(-torch.tensor(score, dtype=torch.float)).numpy()

def _update_scores(scores, key, normal_score, abnormal_score):
    """Update score dictionaries with computed values"""
    final_score = float(normal_score / (normal_score + abnormal_score))
    scores['scores'][key] = DEFAULT_ABNORMAL_SCORE if final_score == DEFAULT_NORMAL_SCORE else final_score
    scores['thresholds'][key] = scores['scores'][key]
    scores['normal'][key] = normal_score
    scores['abnormal'][key] = abnormal_score
    return scores

def _handle_missing_sequence(scores, key, normal_score):
    """Handle sequences missing from abnormal data"""
    scores['thresholds'][key] = DEFAULT_NORMAL_SCORE
    scores['scores'][key] = float(normal_score) if unittest_val(key) else DEFAULT_NORMAL_SCORE
    return scores

def _is_valid_root_sequence(key, scores, template_miner):
    """Validate root sequence criteria"""
    dst_template = id_to_blueprint(int(key.split('_')[1]), template_miner)
    if any(metric in dst_template for metric in ['Cpu', 'Network', 'Memory']):
        return True
        
    for other_key in scores:
        src = int(key.split('_')[0])
        other_dst = int(other_key.split('_')[1])
        if src == other_dst and scores[key] <= scores[other_key]:
            return False
    return True

def _create_result_entry(key, score, dep_info, depth_dict, alarms):
    """Create result entry with metadata"""
    pod = dep_info['pod'] or "frontend-579b9bff58-t2dbm"
    depth = dep_info['depth'] if pod != "frontend-579b9bff58-t2dbm" else 1
   
    depth_dict[pod] = max(depth_dict.get(pod, 0), depth)
   
    result = {
        "records": key,
        "score": score,
        "deepth": depth,
        "duration": dep_info['duration'],
        "pod": pod
    }
   
    for alarm in alarms:
        if alarm["pod"] == pod:
            result["resource"] = alarm["alarm"][0]["metric_type"]
            break
           
    return result

def _get_filtered_indices(results, alarms, depth_dict):
    """Get indices of results to be filtered out"""
    filtered = set()
    for alarm in alarms:
        if alarm["pod"] not in depth_dict:
            continue
           
        max_depth = depth_dict[alarm["pod"]]
        found_valid = False
       
        for idx, result in enumerate(results):
            if "resource" in result and result["pod"] == alarm["pod"]:
                if result["resource"] == alarm["alarm"][0]["metric_type"]:
                    if max_depth > result["deepth"] or (max_depth == result["deepth"] and found_valid):
                        filtered.add(idx)
                    else:
                        found_valid = True
                       
    return filtered


def analyze_sequences_contamination(normal_sequences, abnormal_time, dataset,
                                  template_miner, normal_sequences_gnn,
                                  dependency_scores, contamination):
    """
    Analyze sequences with contamination-based threshold
    """
    # Get abnormal sequences
    abnormal_data = _get_abnormal_sequences(abnormal_time, dataset, template_miner)
    abnormal_sequence_dict, alarm_list, abnormal_sequence_dict_gnn, abnormal_metrics_gnn = abnormal_data
   
    # Calculate sequence scores
    score_data = _calculate_sequence_scores(
        normal_sequences, abnormal_sequence_dict,
        normal_sequences_gnn, abnormal_sequence_dict_gnn
    )
   
    # Log sequence analysis results 
    _log_sequence_analysis(score_data, abnormal_metrics_gnn, alarm_list, template_miner)
   
    # Filter sequences based on contamination threshold
    threshold = sequences_threshold_contamination(score_data['thresholds'], contamination)
    filtered_scores = {k:v for k,v in score_data['scores'].items() if float(v) >= threshold}
   
    # Filter root sequences
    root_sequences = _filter_root_sequences(filtered_scores, template_miner)
   
    # Generate results with depth and pod information
    results = _generate_results(root_sequences, dependency_scores, alarm_list)
   
    # Process and filter alarms
    processed_results = _process_alarms(results, alarm_list)
   
    return sorted(processed_results, key=custom_sort_key)


# Define the custom sorting key function
def custom_sort_key(item):
    return (-item['score'], -item['deepth'])

def unittest_val(key):
    return False #Testing only

def unittest_val2(key):
    return False #Testing only

def rca_contamination(normal_times, fault_files, data_path, dataset, template_miner, contamination):
    """
    Perform root cause analysis with contamination-based threshold

    Args:
        normal_times: List of timestamps for normal behavior
        fault_files: List of fault injection files
        dataset: Dataset name
        template_miner: Log template mining instance
        contamination: Contamination rate for anomaly detection

    Returns:
        dict: Accuracy metrics at different thresholds
    """
    fault_counter = 0
    rankings = []

    # Process each fault injection file
    for normal_time, fault_file in zip(normal_times, fault_files):
        # Get normal behavior sequences
        sequences = get_sequence(
            normal_time,
            dataset,
            data_path,
            template_miner,
            mode='train'
        )
        normal_sequences, record_sequences, alarm_list, normal_sequences_gnn, gnn_metrics = sequences
       
        # Calculate record metrics
        metrics = {
            'impact': calculate_record_impact(record_sequences),
            'depths': calculate_all_record_depths(record_sequences),
            'durations': calculate_all_path_durations(record_sequences)
        }
       
        # Generate dependency scores
        record_depths, record_pods, timestamps = metrics['depths']
        dep_scores = calculate_final_scores(
            metrics['impact'],
            record_depths,
            record_pods,
            timestamps,
            metrics['durations']
        )

        # Load fault and root cause data
        with open(fault_file) as f:
            fault_data = json.load(f)
           
        root_cause_path = f"{data_path}/root_cause_{dataset}.json"
        with open(root_cause_path) as f:
            root_causes = json.load(f)

        # Process each fault with contamination-based ranking
        for hour_data in fault_data.values():
            for fault in hour_data:
                try:
                    abnormal_time = calculate_analysis_time(fault["inject_time"])
                   
                    # Get ranked sequences with contamination threshold
                    results = analyze_sequences_contamination(
                        normal_sequences,
                        abnormal_time,
                        dataset,
                        template_miner,
                        normal_sequences_gnn,
                        dep_scores,
                        contamination
                    )

                    result_list = results
                   
                    # Calculate ranking
                    rank = determine_ranking(
                        fault,
                        result_list,
                        root_causes,
                        template_miner
                    )
                   
                    if rank:
                        rankings.append(rank)
                       
                    log_detailed_results(result_list, template_miner)
                    fault_counter += 1
                       
                except Exception as e:
                    logger.error(f"Error processing fault: {str(e)}")
                    raise

    # Calculate and log metrics
    metrics = calculate_accuracy_metrics(rankings, fault_counter)

    logger.warning("Analysis Configuration:")
    logger.warning(f"Dataset: {dataset}")
    logger.warning(f"Contamination Rate: {contamination}")
    logger.warning(f"Total Faults Detected: {fault_counter}")
    logger.warning("Model Performance:")
    for metric, value in metrics.items():
        logger.warning(f"AS{metric}: {value}")

    return metrics

def rca_epoch(normal_times, fault_files, data_path, dataset, template_miner, epoch):
    """
    Perform root cause analysis with specific epoch training
   
    Args:
        normal_times: List of timestamps for normal behavior
        fault_files: List of fault injection files
        dataset: Dataset name
        template_miner: Log template mining instance
        epoch: Training epoch count
   
    Returns:
        dict: Accuracy metrics at different thresholds
    """
    fault_counter = 0
    rankings = []
   
    for normal_time, fault_file in zip(normal_times, fault_files):
        sequences = get_sequence(
            normal_time,
            dataset,
            data_path,
            template_miner,
            mode='train',
            train_epoch=epoch
        )
        normal_sequences, record_sequences, alarm_list, gnn_sequences, gnn_metrics = sequences
       
        metrics = {
            'impact': calculate_record_impact(record_sequences),
            'depths': calculate_all_record_depths(record_sequences),
            'durations': calculate_all_path_durations(record_sequences)
        }
       
        record_depths, record_pods, timestamps = metrics['depths']
        dep_scores = calculate_final_scores(
            metrics['impact'],
            record_depths,
            record_pods,
            timestamps,
            metrics['durations']
        )

        with open(fault_file) as f:
            fault_data = json.load(f)
           
        root_cause_path = f"{data_path}/root_cause_{dataset}.json"
        with open(root_cause_path) as f:
            root_causes = json.load(f)

        for faults in fault_data.values():
            rankings.extend(
                process_faults(
                    faults,
                    normal_sequences,
                    record_sequences,
                    gnn_sequences,
                    gnn_metrics,
                    dep_scores,
                    root_causes,
                    template_miner,
                    dataset
                )
            )
           
            fault_counter += len(faults)

    metrics = calculate_accuracy_metrics(rankings, fault_counter)
   
    logger.warning(f"Dataset Analysis: {dataset}")
    logger.warning(f"Training Epoch: {epoch}")
    logger.warning(f"Total Faults Detected: {fault_counter}")
    logger.warning("Accuracy Scores:")
    for metric, value in metrics.items():
        logger.warning(f"AS{metric}: {value}")
   
    return metrics

def rca(normal_times, fault_files, data_path, dataset, template_miner):
    """
    Perform root cause analysis on system records and calculate accuracy metrics
   
    Args:
        normal_times: List of timestamps representing normal system behavior
        fault_files: List of files containing fault injection data
        data_path: Path to construction data
        dataset: Dataset name (OnlineBoutique/TrainTicket)
        template_miner: Log template mining instance
   
    Returns:
        dict: Accuracy metrics at different thresholds
    """
    fault_counter = 0
    rankings = []
   
    for normal_time, fault_file in zip(normal_times, fault_files):
        logger.info(f"Normal Time: {normal_time}")
        sequences = get_sequence(normal_time, dataset, data_path, template_miner, mode='train')
        normal_sequences, record_sequences, alarm_list, normal_sequences_gnn, gnn_metrics = sequences
       
        metrics = {
            'impact': calculate_record_impact(record_sequences),
            'depths': calculate_all_record_depths(record_sequences),
            'durations': calculate_all_path_durations(record_sequences)
        }
       
        record_depths, record_pods, timestamps = metrics['depths']
        dep_scores = calculate_final_scores(
            metrics['impact'],
            record_depths,
            record_pods,
            timestamps,
            metrics['durations']
        )

        with open(fault_file) as f:
            fault_data = json.load(f)
           
        root_cause_path = f"{data_path}/root_cause_{dataset}.json"
        with open(root_cause_path) as f:
            root_causes = json.load(f)

        for faults in fault_data.values():
            logger.info(f"Faults: {faults}")
            rankings.extend(
                process_faults(
                    faults,
                    normal_sequences,
                    normal_sequences_gnn,
                    dep_scores,
                    root_causes,
                    template_miner,
                    dataset
                )
            )
            fault_counter += len(faults)

    metrics = calculate_accuracy_metrics(rankings, fault_counter)
    log_results(metrics, dataset, fault_counter,rankings)
   
    return metrics



def rca_explain(normal_times, explain_time, fault_files, data_path, dataset, template_miner):
    """
    Perform root cause analysis on system records and calculate accuracy metrics
   
    Args:
        normal_times: List of timestamps representing normal system behavior
        fault_files: List of files containing fault injection data
        data_path: Path to construction data
        dataset: Dataset name (OnlineBoutique/TrainTicket)
        template_miner: Log template mining instance
   
    Returns:
        dict: Accuracy metrics at different thresholds
    """
    fault_counter = 0
    rankings = []
   
    for normal_time, fault_file in zip(normal_times, fault_files):

        logger.info(f"Normal Time: {normal_time}")
        sequences = get_sequence(normal_time, dataset, data_path, template_miner, mode='train')
        normal_sequences, record_sequences, alarm_list, normal_sequences_gnn, gnn_metrics = sequences
       
        metrics = {
            'impact': calculate_record_impact(record_sequences),
            'depths': calculate_all_record_depths(record_sequences),
            'durations': calculate_all_path_durations(record_sequences)
        }
       
        record_depths, record_pods, timestamps = metrics['depths']
        dep_scores = calculate_final_scores(
            metrics['impact'],
            record_depths,
            record_pods,
            timestamps,
            metrics['durations']
        )

        with open(fault_file) as f:
            fault_data = json.load(f)

        # Filter and reassign to fault_data
        fault_data = {hour: [event for event in events if event["inject_time"] == explain_time] 
                    for hour, events in fault_data.items() if any(event["inject_time"] == explain_time for event in events)}


        root_cause_path = f"{data_path}/root_cause_{dataset}.json"
        with open(root_cause_path) as f:
            root_causes = json.load(f)

        for faults in fault_data.values():
            logger.info(f"Faults: {faults}")
            rankings.extend(
                process_faults(
                    faults,
                    normal_sequences,
                    normal_sequences_gnn,
                    dep_scores,
                    root_causes,
                    template_miner,
                    dataset
                )
            )
            fault_counter += len(faults)

    if fault_counter>0:
        metrics = calculate_accuracy_metrics(rankings, fault_counter)
        log_results(metrics, dataset, fault_counter,rankings)
    else:
        metrics = {
            "@1": "0%",
            "@3": "0%", 
            "@5": "0%"
        }
        logger.info(f"results: {metrics}")
    return metrics

def save_results_to_csv(results, timestamp, dataset):
    """
    Save analysis results to CSV file
    """
    csv_file = f"{ANALYSIS_DIR}/{dataset}/result.csv"
    rows = []
    
    for result in results:
        source, target = result['records'].split('_')
        row = {
            'timestamp': timestamp,
            'source': source,
            'target': target,
            'score': result['score'],
            'deepth': result['deepth'],
            'pod': result['pod'],
            'resource': result.get('resource', '')  # Handle optional resource field
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)


def save_rank_to_csv(rank, abnormal_time, dataset, root_causes, fault):
    """
    Save analysis ranking to CSV file
    """
    csv_file = f"{ANALYSIS_DIR}/{dataset}/rank.csv"
    rows = []
    row = {
        'timestamp': abnormal_time,
        'fault': fault,
        'root_causes': root_causes,
        'rank': rank
    }
    rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)


def process_faults(faults, normal_sequences, normal_sequences_gnn, dep_scores, root_causes, template_miner, dataset):
    """Process individual faults and determine rankings"""
    rankings = []
   
    for fault in faults:
        try:
            abnormal_time = calculate_analysis_time(fault["inject_time"])
           
            results = analyze_sequences(
                normal_sequences,
                abnormal_time,
                dataset,
                template_miner,
                normal_sequences_gnn,
                dep_scores
            )
            
            result_list = results

            save_results_to_csv(results, abnormal_time, dataset)

            logger.info(f"Result before ranking: {result_list}")
           
            rank = determine_ranking(
                fault,
                result_list,
                root_causes,
                template_miner
            )

            

            logger.info(f"Rank: {rank}")
           
            if rank:
                rankings.append(rank)
                save_rank_to_csv(rank, abnormal_time, dataset, root_causes, fault)
               
            log_detailed_results(result_list, template_miner)
               
        except Exception as e:
            logger.error(f"Error processing fault: {str(e)}")
            raise

    return rankings


def calculate_accuracy_metrics(rankings, total_faults):
    """Calculate accuracy metrics from rankings"""
    counts = {
        'top1': sum(1 for r in rankings if r == 1),
        'top3': sum(1 for r in rankings if r <= 3),
        'top5': sum(1 for r in rankings if r <= 5)
    }
    
    return {
        f"@{k[-1]}": f"{(v/total_faults * 100):.2f}%" 
        for k,v in counts.items()
    }

def calculate_analysis_time(inject_time):
    """
    Calculate analysis time point based on injection time
    Adds 2 minutes to injection time for analysis window
    """
    minute = int(inject_time.split(":")[1]) + 2
    hour_str = inject_time.split(" ")[1].split(":")[0]
    hour = int(hour_str)
    
    if minute >= 60:
        if hour < 9:
            return f"{inject_time.split(' ')[0]} 0{hour+1}:0{minute-60}"
        return f"{inject_time.split(' ')[0]} {hour+1}:0{minute-60}"
    elif minute < 10:
        return f"{inject_time.split(':')[0]}:0{minute}"
    return f"{inject_time.split(':')[0]}:{minute}"

def determine_ranking(fault, result_list, root_causes, template_miner):
    """
    Determine ranking position of actual root cause in results
    Returns ranking position if found
    """
    topk = 1
    inject_service = fault["inject_pod"].rsplit('-', 1)[0].rsplit('-', 1)[0]
    root_cause = root_causes[inject_service][fault["inject_type"]].split("_")
    logger.info(f"Root Cause: {root_cause}")
    logger.info(f"Injected Service: {inject_service}")
    
    for i, result in enumerate(result_list):
        if len(root_cause) == 1:
            if _check_single_cause(result, root_cause[0], fault["inject_pod"]):
                return topk
        elif len(root_cause) == 2:
            if _check_double_cause(result, root_cause, fault["inject_pod"], template_miner):
                return topk
        
        if i > 0 and not _same_score(result_list[i-1], result):
            topk += 1
    return None

def log_results(metrics, dataset, fault_count,rankings):
    """Log analysis results and metrics"""
    logger.info(f"Analysis Results - {dataset}")
    logger.info(f"Total Analyzed Faults: {fault_count}")
    logger.info(f"Ranking: {rankings}")
    logger.info("Accuracy Metrics:")
    for metric, value in metrics.items():
        logger.info(f"AS{metric}: {value}")

def log_detailed_results(results, template_miner, max_results=10):
    """Log detailed analysis results"""
    for result in results[:max_results]:
        source = id_to_blueprint(int(result["records"].split("_")[0]), template_miner)
        target = id_to_blueprint(int(result["records"].split("_")[1]), template_miner)
       
        log_msg = (f"Source: {source}, Target: {target}, "
                  f"Score: {result['score']}, Depth: {result['deepth']}, "
                  f"Pod: {result['pod']}")
       
        if "resource" in result:
            log_msg += f", Resource: {result['resource']}"
           
        logger.info(log_msg)

def _check_single_cause(result, root_cause, inject_pod):
    """
    Check if result matches single root cause criteria
    """
    if "resource" in result:
        return (str(root_cause) in str(result["resource"]) and 
                str(inject_pod) in str(result["pod"]))
    return False

def _check_double_cause(result, root_cause, inject_pod, template_miner):
    """
    Check if result matches double root cause criteria
    """
    records = result["records"].split("_")
    source_template = id_to_blueprint(int(records[0]), template_miner)
    target_template = id_to_blueprint(int(records[1]), template_miner)
   
    return (root_cause[0] in source_template and
            root_cause[1] in target_template and
            str(inject_pod) in str(result["pod"]))

def _same_score(prev_result, curr_result):
    """
    Check if two results have the same score and depth
    """
    return (prev_result["score"] == curr_result["score"] and
            prev_result["deepth"] == curr_result["deepth"])
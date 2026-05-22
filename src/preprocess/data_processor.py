import pandas as pd
import numpy as np
import os
import json
from src.preprocess.log_parser import *
import re
import json
import statistics
from src.config import *
import concurrent.futures
from src.log.log_handler import get_logger

logger = get_logger(__name__)

class TraceObj:
    """Represents a trace containing multiple spans"""
    def __init__(self, traceid):
        self.traceid = traceid
        self.spans = []

    def sort_spans(self):
        """Sort spans by their first record's timestamp"""
        self.spans.sort(key=lambda k: k.records[0].timestamp)

    def append_spans(self, span):
        """Add a span to the trace"""
        self.spans.append(span)

    def show_all_spans(self):
        """Display all spans and their records"""
        logger.debug(f"traceid: {self.traceid}")
        for span in self.spans:
            span.show_all_records()

class SpanObj:
    """Represents a span containing multiple records"""
    def __init__(self, spanid, parentid, pod):
        self.spanid = spanid
        self.parentid = parentid
        self.pod = pod
        self.records = []

    def sort_records(self):
        """Sort records by timestamp"""
        self.records.sort(key=lambda k: k.timestamp)

    def append_record(self, record):
        """Add a record to the span"""
        self.records.append(record)

    def new_timestamp(self):
        """Generate timestamp for new record"""
        return self.records[-1].timestamp - 1

    def show_all_records(self):
        """Display span details and all records"""
        logger.debug("%s,%s,%s", self.spanid, self.parentid, self.pod)
        for record in self.records:
            record.show_record()
        logger.debug("")

class Record:
    """Represents a single record in a span"""
    def __init__(self, record, pod, dataset, timestamp=None, spanid=None, parentid=None):
        self.record = record
        self.pod = pod
        self.timestamp = timestamp
        self.spanid = spanid
        self.parentid = parentid
        self.dataset = dataset

    def show_record(self):
        """Display record details"""
        logger.debug("Record: %s, %s, %s", 
                   self.timestamp, 
                   self.spanid, 
                   self.record)
class RecordSequence:
    """Represents a directed graph of records"""
    def __init__(self, log_template):
        self.adjacency_dict = {}
        self.nodes = set()
        self.unique_pairs = set()
        self.sequence_counts = {}
        self.log_template = log_template

    def add_edge(self, node1, node2):
        """Add directed edge between two nodes"""
        if node1 not in self.adjacency_dict:
            self.adjacency_dict[node1] = []
        self.adjacency_dict[node1].append(node2)
        self.nodes.add(node1.record)
        self.nodes.add(node2.record)

    def remove_edge(self, node1, node2):
        """Remove edge between two nodes"""
        self.adjacency_dict[node1].remove(node2)

    def print_adj_list(self):
        """Display adjacency list in a structured format"""
        def format_node(node, connections):
            return f"Node ID: {node}\nConnections: {', '.join(map(str, connections))}\n"
            
        formatted_output = [format_node(k, v) for k, v in self.adjacency_dict.items()]
        print("\nSequence Structure:")
        print("-" * 50)
        print("\n".join(formatted_output))
        print("-" * 50)


    def show_sequences(self):
        """Visualize sequences structure with detailed template information"""
        def get_template_info(record):
            template = id_to_blueprint(record, self.log_template)
            return f"ID: {record} | Template: {template}"
        
        logger.debug("Sequence Template Analysis:")
        logger.debug("=" * 80)
        
        for source, targets in self.adjacency_dict.items():
            logger.debug("\nSource Node:")
            logger.debug("|- %s", get_template_info(source.record))
            
            if targets:
                logger.debug("L Connected to:")
                for idx, target in enumerate(targets, 1):
                    prefix = "   |-" if idx < len(targets) else "   L"
                    logger.debug("%s %s", prefix, get_template_info(target.record))
                    
        logger.debug("=" * 80)

    def calc_pod_depth(self, target_record):
        """Calculate depth and find pod for target record"""
        def check_template_conditions(record_key):
            template = id_to_blueprint(record_key.record, self.log_template)
            return "start" in template and "TraceID" not in template
        
        def process_record_match(current_key, matched_item, current_depth):
            next_record = current_key.record
            pod_value = matched_item.pod if current_depth == 0 else ""
            depth_increment = 1 if check_template_conditions(current_key) else 0
            return next_record, pod_value, depth_increment
        
        current_record = target_record
        total_depth = 0
        target_pod = ""
        
        while current_record:
            record_found = False
            
            for source_record, target_records in self.adjacency_dict.items():
                matching_records = [r for r in target_records if r.record == current_record]
                
                if matching_records:
                    current_record, pod, depth_inc = process_record_match(
                        source_record, matching_records[0], total_depth
                    )
                    total_depth += depth_inc
                    target_pod = pod if pod else target_pod
                    record_found = True
                    break
                    
            if not record_found:
                break
                
        return total_depth, target_pod

    def extract_sequences(self):
        """
        Extract and count unique record sequences from the sequence
        
        Returns:
            dict: Mapping of record pairs to their occurrence count
            {
                "record1_record2": count,
                ...
            }
        """
        
        for source, targets in self.adjacency_dict.items():
            for target in targets:
                sequence_key = f"{source.record}_{target.record}"
                if sequence_key not in self.unique_pairs:
                    self.sequence_counts[sequence_key] = 1
                    self.unique_pairs.add(sequence_key)
                else:
                    self.sequence_counts[sequence_key] += 1
                    
        return self.sequence_counts


def construct_trace_record_sequence(trace_reader, log_reader, trace_id, alarm_list, dataset, log_template):
    """
    Construct a complete trace sequence including spans, logs, and metric alarms
    """
    trace = TraceObj(f"StartTraceId is {trace_id}")
    log_span_ids = log_reader.index.tolist()
    
    try:
        spans = _get_trace_spans(trace_reader, trace_id)
        if len(spans['SpanID']) > 0:
            _process_trace_spans(spans, log_reader, log_span_ids, alarm_list, dataset, log_template, trace)
            
    except Exception as e:
        pass
        #logger.error("Error processing trace: %s", e)
        

    return trace

def _get_trace_spans(trace_reader, trace_id):
    """Extract spans for a given trace"""
    try:
        return trace_reader.loc[[trace_id], [
            'SpanID', 'ParentID', 'PodName',
            'StartTimeUnixNano', 'EndTimeUnixNano', 'OperationName'
        ]]
    except Exception as e:
        logger.error("Error fetching spans for trace: %s", e)
        raise

#def _process_trace_spans(spans, log_reader, log_span_ids, alarm_list, dataset, log_template, trace):
#    """Process all spans in a trace"""
#    for _, span_data in spans.iterrows():
#        span = _create_span(span_data)
#        
#        _add_operation_records(span, span_data, dataset, log_template)
#        _add_log_records(span, span_data, log_reader, log_span_ids, dataset, log_template)
#        _add_alarm_records(span, alarm_list, span_data, dataset, log_template)
#        
#        span.sort_records()
#        trace.append_spans(span)
#        trace.show_all_spans()


#def _process_trace_spans(spans, log_reader, log_span_ids, alarm_list, dataset, log_template, trace):
#    """Process all spans in a trace with ordered event handling"""
#    for _, span_data in spans.iterrows():
#        span = _create_span(span_data)
#        
#        # Add events in specific order
#        _add_operation_start_record(span, span_data, dataset, log_template)
#        _add_log_records(span, span_data, log_reader, log_span_ids, dataset, log_template)
#        _add_alarm_records(span, alarm_list, span_data, dataset, log_template)
#        _add_operation_end_record(span, span_data, dataset, log_template)
#        
#        span.sort_records()
#        trace.append_spans(span)
#        trace.show_all_spans()
#        
#    trace.sort_spans()

def _process_trace_spans(spans, log_reader, log_span_ids, alarm_list, dataset, log_template, trace):
    """Process all spans in a trace"""
    for _, span_data in spans.iterrows():
        span = SpanObj(
            spanid=span_data['SpanID'],
            parentid=span_data['ParentID'],
            pod=span_data['PodName']
        )

        # Extract service name exactly as original
        service = span_data['PodName'].rsplit('-', 1)[0]
        service = service.rsplit('-', 1)[0]

        # Add start record with matching template
        start_record = f"{service} {span_data['OperationName']} start"
        start_timestamp = np.ceil(span_data['StartTimeUnixNano']).astype(int)
        span.append_record(Record(
            timestamp=start_timestamp,
            record=log_blueprint_parse(log=start_record, pod=span_data['PodName'], log_template=log_template),
            pod=span_data['PodName'],
            dataset=dataset,
            spanid=span_data['SpanID'],
            parentid=span_data['ParentID']
        ))

        end_timestamp = np.ceil(span_data['EndTimeUnixNano']).astype(int)

        # Process logs in same order as original
        if span_data['SpanID'] in log_span_ids:
            logs = log_reader.loc[[span_data['SpanID']], ['TimeUnixNano', 'Log']]
            for _, log_data in logs.iterrows():
                timestamp = np.ceil(log_data['TimeUnixNano']).astype(int)
                if timestamp - end_timestamp > 0:
                    end_timestamp = timestamp + 1
                span.append_record(Record(
                    timestamp=timestamp,
                    record=log_blueprint_parse(log=log_data['Log'], pod=span_data['PodName'], log_template=log_template),
                    pod=span_data['PodName'],
                    dataset=dataset,
                    spanid=span_data['SpanID'],
                    parentid=span_data['ParentID']
                ))

        # Add end record
        end_record = f"{service} {span_data['OperationName']} end"
        span.append_record(Record(
            timestamp=end_timestamp,
            record=log_blueprint_parse(log=end_record, pod=span_data['PodName'], log_template=log_template),
            pod=span_data['PodName'],
            dataset=dataset,
            spanid=span_data['SpanID'],
            parentid=span_data['ParentID']
        ))

        # Add alarms maintaining original order
        if dataset == "OnlineBoutique" and len(span.records) > 2:
            _process_alarms(span, alarm_list, span_data, dataset, log_template)
        elif dataset == "TrainTicket":
            _process_alarms(span, alarm_list, span_data, dataset, log_template)

        span.sort_records()
        trace.append_spans(span)
        #trace.show_all_spans()

    trace.sort_spans()
    return trace

def _process_alarms(span, alarm_list, span_data, dataset, log_template):
    """Process alarms in original order"""
    for alarm in alarm_list:
        if alarm["pod"] == span.pod:
            for idx, alarm_data in enumerate(alarm["alarm"]):
                span.append_record(Record(
                    timestamp=np.ceil(span_data['StartTimeUnixNano']).astype(int) + idx + 1,
                    record=log_blueprint_parse(log=alarm_data["metric_type"], pod="alarm", log_template=log_template),
                    pod=span.pod,
                    dataset=dataset,
                    spanid=span_data['SpanID'],
                    parentid=span_data['ParentID']
                ))
            break



def _add_operation_start_record(span, span_data, dataset, log_template):
    """Add operation start record"""
    service = _extract_service_name(span_data['PodName'])
    start_record = f"{service} {span_data['OperationName']} start"
    
    span.append_record(Record(
        timestamp=np.ceil(span_data['StartTimeUnixNano']).astype(int),
        record=log_blueprint_parse(log=start_record, pod=span_data['PodName'], log_template=log_template),
        pod=span_data['PodName'],
        dataset=dataset,
        spanid=span_data['SpanID'],
        parentid=span_data['ParentID']
    ))

def _add_operation_end_record(span, span_data, dataset, log_template):
    """Add operation end record"""
    service = _extract_service_name(span_data['PodName'])
    end_record = f"{service} {span_data['OperationName']} end"
    
    span.append_record(Record(
        timestamp=np.ceil(span_data['EndTimeUnixNano']).astype(int),
        record=log_blueprint_parse(log=end_record, pod=span_data['PodName'], log_template=log_template),
        pod=span_data['PodName'],
        dataset=dataset,
        spanid=span_data['SpanID'],
        parentid=span_data['ParentID']
    ))

def _create_span(span_data):
    """Create a new span object"""
    return SpanObj(
        spanid=span_data['SpanID'],
        parentid=span_data['ParentID'],
        pod=span_data['PodName']
    )

def _add_operation_records(span, span_data, dataset, log_template):
    """Add operation start/end records to span"""
    service = _extract_service_name(span_data['PodName'])
    operation = span_data['OperationName']
    
    start_record = f"{service} {operation} start"
    end_record = f"{service} {operation} end"
    
    span.append_record(Record(
        timestamp=np.ceil(span_data['StartTimeUnixNano']).astype(int),
        record=log_blueprint_parse(log=start_record, pod=span_data['PodName'], log_template=log_template),
        pod=span_data['PodName'],
        dataset=dataset,
        spanid=span_data['SpanID'],
        parentid=span_data['ParentID']
    ))
    
    span.append_record(Record(
        timestamp=np.ceil(span_data['EndTimeUnixNano']).astype(int),
        record=log_blueprint_parse(log=end_record, pod=span_data['PodName'], log_template=log_template),
        pod=span_data['PodName'],
        dataset=dataset,
        spanid=span_data['SpanID'],
        parentid=span_data['ParentID']
    ))

def _add_log_records(span, span_data, log_reader, log_span_ids, dataset, log_template):
    """Add log records to span"""
    if span_data['SpanID'] in log_span_ids:
        try:
            logs = log_reader.loc[[span_data['SpanID']], ['TimeUnixNano', 'Log']]
            for _, log_data in logs.iterrows():
                span.append_record(Record(
                    timestamp=np.ceil(log_data['TimeUnixNano']).astype(int),
                    record=log_blueprint_parse(log=log_data['Log'], pod=span_data['PodName'], log_template=log_template),
                    pod=span_data['PodName'],
                    dataset=dataset,
                    spanid=span_data['SpanID'],
                    parentid=span_data['ParentID']
                ))
        except Exception as e:
            logger.error("Error processing logs: %s", e)

def _add_alarm_records(span, alarm_list, span_data, dataset, log_template):
    """Add alarm records to span"""
    if dataset == "OnlineBoutique" and len(span.records) > 2:
        _process_alarms(span, alarm_list, span_data, dataset, log_template)
    elif dataset == "TrainTicket":
        _process_alarms(span, alarm_list, span_data, dataset, log_template)

def _process_alarms(span, alarm_list, span_data, dataset, log_template):
    """Process and add alarm records"""
    for alarm in alarm_list:
        if alarm["pod"] == span.pod:
            for idx, alarm_data in enumerate(alarm["alarm"]):
                span.append_record(Record(
                    timestamp=np.ceil(span_data['StartTimeUnixNano']).astype(int) + idx + 1,
                    record=log_blueprint_parse(log=alarm_data["metric_type"], pod="alarm", log_template=log_template),
                    pod=span.pod,
                    dataset=dataset,
                    spanid=span_data['SpanID'],
                    parentid=span_data['ParentID']
                ))
            break

def _extract_service_name(pod_name):
    """Extract service name from pod name"""
    service = pod_name.rsplit('-', 1)[0]
    return service.rsplit('-', 1)[0]

def construct_record_chain(alarm_list, trace, log_template):
    """
    Construct sequential record chains from trace and alarm data
    
    Args:
        alarm_list: List of metric alarms
        trace: Trace object containing spans and records
        log_template: Template parser for logs
        
    Returns:
        tuple: (pod_record_chain, record_chain)
            - pod_record_chain: List of pod-record pairs
            - record_chain: List of record IDs
    """
    record_sequence = _build_record_sequence(trace)
    pod_chain, record_chain = _convert_to_chains(record_sequence)
    
    if alarm_list:
        pod_chain, record_chain = _insert_alarms(
            alarm_list, pod_chain, record_chain, log_template
        )
        
    return pod_chain, record_chain

def _build_record_sequence(trace):
    """Build ordered sequence of records from trace"""
    sequence = []
    
    for span in trace.spans:
        if span.parentid == "root":
            sequence.extend(span.records)
        else:
            _insert_span_records(span, sequence)
            
    return sequence

def _insert_span_records(span, sequence):
    """Insert span records in correct positions"""
    parent_indices = _find_parent_indices(span, sequence)
    
    for record in reversed(span.records):
        for index in parent_indices:
            if record.timestamp > sequence[index].timestamp:
                sequence.insert(index + 1, record)
                break

def _find_parent_indices(span, sequence):
    """Find indices for parent-related records"""
    parent_direct = [
        i for i, record in enumerate(sequence)
        if record.spanid == span.parentid
    ]
    
    parent_indirect = [
        i for i, record in enumerate(sequence)
        if record.parentid == span.parentid and record.pod == span.pod
    ]
    
    indices = parent_direct + parent_indirect
    return sorted(indices, reverse=True)

def _convert_to_chains(record_sequence):
    """Convert record sequence to pod and record chains"""
    pod_chain = [
        f"{record.pod}_{record.record}"
        for record in record_sequence
    ]
    
    record_chain = [
        record.record
        for record in record_sequence
    ]
    
    return pod_chain, record_chain

def _insert_alarms(alarm_list, pod_chain, record_chain, log_template):
    """Insert alarms at appropriate positions in chains"""
    for alarm in alarm_list:
        pod = alarm['pod']
        
        for i, pod_record in enumerate(pod_chain):
            if pod in pod_record:
                for alarm_data in alarm['alarm']:
                    record_id = log_blueprint_parse(
                        log=alarm_data['metric_type'],
                        pod="alarm",
                        log_template=log_template
                    )
                    
                    pod_record = f"{pod}{record_id}"
                    pod_chain.insert(i, pod_record)
                    record_chain.insert(i, record_id)
                break
                
    return pod_chain, record_chain


def construct_record_sequence(trace, log_template):
    """
    Construct directed graph representation of record sequences
    
    Args:
        trace: Trace object containing spans and records
        log_template: Template parser for logs
        
    Returns:
        RecordSequence: Directed graph of record sequences
    """
    sequence = RecordSequence(log_template)
    
    _add_intra_span_edges(trace, sequence)
    _add_inter_span_edges(trace, sequence)
    
    return sequence

def _add_intra_span_edges(trace, sequence):
    """Add edges between records within each span"""
    for span in trace.spans:
        for i in range(1, len(span.records)):
            sequence.add_edge(span.records[i-1], span.records[i])

def _add_inter_span_edges(trace, sequence):
    """Add edges between parent and child spans"""
    for child_span in trace.spans:
        for parent_span in trace.spans:
            if parent_span.spanid == child_span.parentid:
                if parent_span.pod == child_span.pod:
                    _connect_same_pod_spans(parent_span, child_span, sequence)
                else:
                    _connect_different_pod_spans(parent_span, child_span, sequence)
                break

def _connect_same_pod_spans(parent_span, child_span, sequence):
    """Connect spans from the same pod based on timestamp"""
    child_start_time = child_span.records[0].timestamp
    
    for i in range(1, len(parent_span.records)):
        if parent_span.records[i].timestamp > child_start_time:
            sequence.add_edge(parent_span.records[i-1], child_span.records[0])
            break

def _connect_different_pod_spans(parent_span, child_span, sequence):
    """Connect spans from different pods"""
    sequence.add_edge(parent_span.records[0], child_span.records[0])


def analyze_span_error_rates(log_reader):
    """
    Calculate error rates for spans based on log analysis
    
    Args:
        log_reader: DataFrame containing span logs
        
    Returns:
        dict: Mapping of span IDs to their error ratios
        {
            span_id: error_ratio,
            ...
        }
    """
    error_rates = {}
    
    for span_id in log_reader.index.unique():
        error_rate = _calculate_span_error_rate(
            log_reader.loc[[span_id], ['TimeUnixNano', 'Log']]
        )
        if error_rate > 0:
            error_rates[span_id] = error_rate
            
    return error_rates

def _calculate_span_error_rate(span_logs):
    """Calculate error rate for a single span"""
    total_logs = len(span_logs)
    if total_logs == 0:
        return 0
        
    error_count = sum(
        1 for _, log_row in span_logs.iterrows()
        if _is_error_log(log_row['Log'])
    )
    
    return error_count / total_logs

def _is_error_log(log_entry):
    """Check if log entry contains error message"""
    try:
        log_message = json.loads(log_entry)['log']
        return "ERROR" in log_message.upper()
    except (json.JSONDecodeError, KeyError):
        return False

def integrate_multimodal_data(trace_file, trace_id_file, log_file, alarm_list, dataset, log_template):
    """
    Integrate traces, logs, and alarms into record sequences
    
    Args:
        trace_file: Path to trace data file
        trace_id_file: Path to trace IDs file
        log_file: Path to log data file
        alarm_list: List of metric alarms
        dataset: Dataset identifier
        log_template: Template parser for logs
        
    Returns:
        tuple: (record_sequences, log_sequences)
    """
    #logger.info(f"Processing alarms: {alarm_list}")
    
    # Load data files
    trace_ids = _load_trace_ids(trace_id_file)
    trace_data = _load_trace_data(trace_file)
    log_data = _load_log_data(log_file)
    
    # Process traces and construct sequences
    log_sequences = _process_traces(trace_ids, trace_data, log_data, alarm_list, dataset, log_template)
    record_sequences = _construct_sequences(log_sequences, log_template)

    #logger.info(f"Record sequences: {record_sequences}")
    #logger.info(f"Log sequences: {log_sequences}")

    
    # Calculate support values
    #for sequence in record_sequences:
    #    sequence.extract_sequences()
        
    return record_sequences, log_sequences

def _load_trace_ids(trace_id_file):
    """Load trace IDs from file"""
    return pd.read_csv(trace_id_file, header=None, engine='c')[0]

def _load_trace_data(trace_file):
    """Load trace data from file"""
    return pd.read_csv(
        trace_file,
        index_col='TraceID',
        usecols=[
            'TraceID', 'SpanID', 'ParentID', 'PodName',
            'StartTimeUnixNano', 'EndTimeUnixNano', 'OperationName'
        ],
        engine='c'
    )

def _load_log_data(log_file):
    """Load log data from file"""
    return pd.read_csv(
        log_file,
        index_col='SpanID',
        usecols=['TimeUnixNano', 'SpanID', 'Log'],
        engine='c'
    )

def _process_traces(trace_ids, trace_data, log_data, alarm_list, dataset, log_template):
    """Process traces using parallel execution"""
    log_sequences = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                construct_trace_record_sequence,
                trace_data,
                log_data,
                trace_id,
                alarm_list,
                dataset,
                log_template
            )
            for trace_id in trace_ids
        }
        
        for future in concurrent.futures.as_completed(futures): 
            if result := future.result():
                log_sequences.append(result)
                

    return log_sequences

def _construct_sequences(log_sequences, log_template):
    """Construct record sequencehs using parallel execution"""
    record_sequences = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                construct_record_sequence,
                sequence,
                log_template
            )
            for sequence in log_sequences
        }
        
        for future in concurrent.futures.as_completed(futures):
            if result := future.result():
                record_sequences.append(result)
                
    return record_sequences

'''
def initialize_log_template(dataset):
    """
    Initialize and configure log template miner for specific dataset
    
    Args:
        dataset: Dataset identifier (e.g., 'OnlineBoutique', 'TrainTicket')
        
    Returns:
        TemplateMiner: Configured template mining instance
    """
    base_path = Path(__file__).parent.resolve()
    template_dir = base_path / 'log_template'
    
    # Configure template miner
    config = _create_miner_config(template_dir, dataset)
    
    # Initialize persistence
    persistence_path = template_dir / f"{dataset}.bin"
    persistence = FilePersistence(persistence_path)
    
    # Create and return template miner
    return TemplateMiner(persistence, config=config)

def _create_miner_config(template_dir, dataset):
    """Create and configure template miner settings"""
    config = TemplateMinerConfig()
    config.load(template_dir / f"drain3_{dataset}.ini")
    config.profiling_enabled = False
    return config

'''

def get_metrics(timestamp, base_dir): #get_metric_with_time
    """
    Get system metrics and log metrics for a specific timestamp
    
    Args:
        timestamp: Target timestamp in format "YYYY-MM-DD HH:MM"
        base_dir: Base directory for data files
        
    Returns:
        tuple: (list of metrics per pod, dictionary of log metrics)
    """
    paths = _get_metric_paths(timestamp, base_dir)
    log_metrics = _process_logs(paths['log'])
    return _collect_metrics(timestamp, paths, base_dir), log_metrics

def _get_metric_paths(timestamp, base_dir):
    """Generate paths for metric data files"""
    date, time = timestamp.split(' ')
    hour, minute = time.split(':')
    
    return {
        'trace': f"{base_dir}/{date}/trace/{hour}_{minute}_trace.csv",
        'metric': f"{base_dir}/{date}/metric/",
        'log': f"{base_dir}/{date}/log/{hour}_{minute}_log.csv"
    }

def _process_logs(log_path):
    """Process logs and calculate success ratios"""
    log_reader = pd.read_csv(log_path, index_col='PodName', usecols=['PodName','Log'], engine='c')
    success_ratios = {}
    
    for pod in log_reader.index.unique():
        pod_logs = log_reader.loc[[pod], ['Log']]
        total_logs = len(pod_logs)
        success_count = _count_successful_logs(pod_logs)
        
        if total_logs > 0:
            success_ratios[pod] = success_count / total_logs
            
    return success_ratios

def _collect_metrics(timestamp, paths, base_dir):
    """Collect all metrics for each pod"""
    metric_types = ['CpuUsageRate(%)', 'MemoryUsageRate(%)']
    metrics = []
    
    for path in os.listdir(paths['metric']):
        if "metric" in path:
            pod_metrics = _process_pod_metrics(
                timestamp, path, paths, metric_types, base_dir
            )
            metrics.extend(pod_metrics)
            
    return metrics

def _process_pod_metrics(timestamp, path, paths, metric_types, base_dir):
    """Process metrics for a single pod"""
    metrics_df = pd.read_csv(os.path.join(paths['metric'], path))
    pod_metrics = []
    
    for idx, row in metrics_df.iterrows():
        if re.search(timestamp, row['Time']):
            metric_entry = {
                "pod": row['PodName'],
                "metrics": _build_metric_list(row, metric_types, paths['trace'], base_dir)
            }
            pod_metrics.append(metric_entry)
            
    return pod_metrics

def _build_metric_list(row, metric_types, trace_path, base_dir):
    """Build list of metrics for a pod"""
    metrics = [
        {"metric_type": metric, "metric_value": row[metric]}
        for metric in metric_types
    ]
    
    network_p90, _ = calc_network_latency(trace_path, row['PodName'], base_dir)
    metrics.append({
        "metric_type": "NetworkP90(ms)",
        "metric_value": network_p90
    })
    
    return metrics

def _count_successful_logs(pod_logs):
    """Count successful log entries"""
    success_count = 0
    for _, log_row in pod_logs.iterrows():
        try:
            log_message = json.loads(log_row['Log'])['log'].lower()
            if ("exception" not in log_message or 
                "nested exception is java.lang.illegalstateexception: no instances available" in log_message):
                success_count += 1
        except json.JSONDecodeError:
            continue
    return success_count

def calc_network_latency(trace_file, pod_name, base_dir):
    """Calculate network metrics for a pod"""
    pod_name = pod_name.strip().lower()
    
    if "front" in pod_name:
        return 10, 10

    #logger.info(f"latency calculated: {_calculate_network_latency(trace_file, pod_name, base_dir)}")
    return _calculate_network_latency(trace_file, pod_name, base_dir)

def _calculate_network_latency(trace_file, pod_name, base_dir):
    """Calculate network latency metrics"""
    #logger.info(f"Calculating network latency for {pod_name}")
    try:
        df = pd.read_csv(trace_file, usecols=['TraceID', 'SpanID', 'ParentID', 'PodName', 'EndTimeUnixNano'])
        df['PodName'] = df['PodName'].str.strip().str.lower()
        
        latencies = _get_pod_latencies(df, pod_name)
        #logger.info(f"latencies: {latencies}")
        #logger.info(f"np.nanpercentile(latencies, 90): {np.nanpercentile(latencies, 90)}")
        #logger.info(f"tatistics.stdev(latencies): {statistics.stdev(latencies)}")
        if len(latencies) > 2:
            return np.nanpercentile(latencies, 90), statistics.stdev(latencies)
        else:
            return 10, 10
            
    except Exception as e:
        #logger.error(f"Error calculating network latency for {pod_name}: {e}")
        service = get_svc(pod_name)
        default_metrics = pd.read_csv(f"{METRIC_THRESHOLD_DATA_DIR}/{service}.csv")
        return float(default_metrics.iloc[0]['NetworkP90(ms)']), 0


def _get_pod_latencies(df, pod_name):
    """Calculate latencies for a pod's spans"""
    pod_reader = df.set_index('PodName')
    parent_span_reader = df.set_index('SpanID')
    latency_list = []

    try:
        pod_spans = pod_reader.loc[[pod_name], ['SpanID', 'ParentID', 'EndTimeUnixNano']]
        
        for span_index in range(len(pod_spans['SpanID'])):
            parent_id = pod_spans['ParentID'].iloc[span_index]
            pod_start_time = int(pod_spans['EndTimeUnixNano'].iloc[span_index])
            
            try:
                parent_pod_span = parent_span_reader.loc[[parent_id], ['PodName', 'EndTimeUnixNano']]
                
                if len(parent_pod_span) > 0:
                    parent_pod_name = parent_pod_span['PodName'].iloc[0]
                    parent_end_time = int(parent_pod_span['EndTimeUnixNano'].iloc[0])
                    
                    if str(parent_pod_name) != str(pod_name):
                        latency = (parent_end_time - pod_start_time) / 1000000
                        latency_list.append(latency)
                        
            except Exception as e:
                #logger.error(f"Error calculating network latency for {pod_name}: {e}")
                continue
                
    except Exception as e:
        #logger.error(f"Error calculating network latency for {pod_name}: {e}")
        raise #Exception(f"Error calculating network latency for {pod_name}: {e}")

    return latency_list

def get_svc(path):
    """Extract service name from pod path"""
    svc = path.rsplit('-', 1)[0]
    return svc.rsplit('-', 1)[0]

def raise_metric(pod, metric_type, metric_value, dataset): 
    """Determine if metric value should trigger alarm"""
    if metric_type in ['CpuUsageRate(%)', 'MemoryUsageRate(%)']:
        return metric_value > METRICS_THRESHOLDS['global_threshold']
    elif metric_type == "NetworkP90(ms)":
        return metric_value > METRICS_THRESHOLDS['network_threshold']
    else:
        thresholds = {'OnlineBoutique': METRICS_THRESHOLDS['service_thresholds']['OnlineBoutique'], 'TrainTicket': METRICS_THRESHOLDS['service_thresholds']['TrainTicket']}
        return metric_value > thresholds.get(dataset, METRICS_THRESHOLDS['service_thresholds']['OnlineBoutique'])

def set_raised_metric(metric_list, dataset, data_path, std_num=6): #generate_alarm
    """Generate alarms based on metric thresholds"""
    alarms = []
    
    for pod_metric in metric_list:
        pod_alarms = _check_pod_alarms(pod_metric, dataset, data_path, std_num)
        if pod_alarms:
            alarms.append(pod_alarms)
            
    return alarms

def _check_pod_alarms(pod_metric, dataset, data_path, std_num):
    """Check metrics for a pod and generate alarms"""
    alarms = []
    for metric in pod_metric['metrics']:
        if raise_metric(pod_metric['pod'], metric['metric_type'], 
                       metric['metric_value'], dataset):
            alarms.append({
                'metric_type': metric['metric_type'],
                'alarm_flag': True
            })
            
    if alarms:
        return {
            'pod': pod_metric['pod'],
            'alarm': alarms
        }
    return None

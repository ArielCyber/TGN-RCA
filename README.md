# TGN-RCA

Official implementation of the paper "Uncovering Microservice Faults: A Temporal Graph Approach to Root Cause Analysis"

## System Requirements

### Hardware
- Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz

### Software
- Anaconda with Python 3.10 (tested)

## Installation


1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
```
Update the DATASET variable in .env to match your dataset name.


## Usage

1. Prepare your dataset in the required format
2. Configure parameters in config.py
3. Run the main analysis pipeline:
```bash
python main.py
```

## Citation
If you use this code for your research, please cite our paper:
```
@TBD
```
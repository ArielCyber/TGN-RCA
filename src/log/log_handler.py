import logging
from functools import lru_cache
import datetime
import os

# Get absolute path to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(ROOT_DIR, 'log', f'system_{datetime.datetime.now().strftime("%Y-%m-%d")}.log')
LOG_LEVEL = logging.DEBUG

class Logger:
    def __init__(self, logname=LOG_PATH, loglevel=LOG_LEVEL, loggername=None):
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(logname), exist_ok=True)
        
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        
        if not self.logger.handlers:
            fh = logging.FileHandler(logname)
            fh.setLevel(loglevel)
            
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            
            formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def getlog(self):
        return self.logger

@lru_cache(maxsize=1)
def get_logger(loggername=None):
    return Logger(LOG_PATH, LOG_LEVEL, loggername).getlog()

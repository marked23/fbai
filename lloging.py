# for single process logging
from hyperparameters import Hyperparameters
import logging
import time

# for single process logging
# @staticmethod
def setup_logging(hp: Hyperparameters):
    log_format = f'[{hp.parameter_set_id}] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("parallel_execution.log"),
            logging.StreamHandler()
        ]
    )

# for multi process logging
# @staticmethod
def setup_logger(hp: Hyperparameters, queue):
    log_format = f'[{hp.parameter_set_id}] %(message)s'
    queue_handler = logging.handlers.QueueHandler(queue)
    formatter = logging.Formatter(log_format)
    queue_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)
    hp.spit = logging.info

# @staticmethod
def listener_configurer(run_path):
    log_format = f'%(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"{run_path}/parallel_execution.log"),
            logging.StreamHandler()
        ]
    )

# @staticmethod
def listener_process(queue, run_path: str):
    listener_configurer(run_path)
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

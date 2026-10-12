import logging
import datetime
import os

# Create a logger object
_logger = None

def init_logger(log_dir="./"):
    os.makedirs(log_dir, exist_ok=True)
    global _logger
    _logger = logging.getLogger('MyLogger')
    _logger.setLevel(logging.INFO)  # Set the default logging level to INFO
    _logger.propagate = False
    if _logger.handlers:
        _logger.handlers.clear()

    # Create a formatter with detailed format including filename and line number
    _formatter = logging.Formatter('%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a file handler and set the level to INFO
    _file_handler = logging.FileHandler(f'{log_dir}output.{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log.txt', mode='w')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_formatter)

    # Create a console handler and set the level to INFO
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_formatter)

    # Add the handlers to the logger
    _logger.addHandler(_file_handler)
    _logger.addHandler(_console_handler)

def get_logger():
    # Model modules obtain their logger at import time. Keep imports usable for
    # inference without creating training log files; init_logger configures this
    # same named logger later when a training run requests file handlers.
    return _logger if _logger is not None else logging.getLogger('MyLogger')

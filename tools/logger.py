import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(name="TTS_Infer", log_dir="logs", log_file="tts_infer.log", level=logging.INFO):
    """
    Setup a logger with console and daily-rotating file handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Formatter
    # Added %(module)s to see which file the log comes from
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Daily Rotation)
    file_handler = TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

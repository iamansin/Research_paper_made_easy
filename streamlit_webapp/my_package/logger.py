import logging
import colorlog
from logging.handlers import RotatingFileHandler


def setup_colored_logging(name:str , log_file_name:str = "application.log", propogation_value:bool= True):
    log_file = log_file_name
    log_level = logging.DEBUG
    max_file_size = 5 * 1024 * 1024  # 5 MB
    backup_count = 3

    # Get the root logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    logger.propagate = propogation_value
    if logger.hasHandlers():
        print("there was a log handler")
        logger.handlers.clear()
        
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - [%(asctime)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s - %(name)s - [%(asctime)s] - - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        

    return logger

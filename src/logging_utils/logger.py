import logging
import os

def get_logger(config):
    logger = logging.getLogger("VehicleSpeedDetection")
    level = getattr(logging, config['logging']['level'].upper(), logging.DEBUG)
    logger.setLevel(level)
    
    # Create console and file handlers
    c_handler = logging.StreamHandler()
    
    # Ensure the logs directory exists before creating the FileHandler
    log_file = config['logging']['log_file']
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create and set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger if they are not already added
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

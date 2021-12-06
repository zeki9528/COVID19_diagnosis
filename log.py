import os
import sys
import logging

def create_logger(log_path, name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s %(name)s] %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
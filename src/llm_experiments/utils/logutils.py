import logging
from pathlib import Path
from sys import stdout


def create_logger(name: str, log_fp: Path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s::%(message)s")

    stdout_handler = logging.StreamHandler(stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_fp:
        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

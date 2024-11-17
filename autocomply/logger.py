import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s|%(name)s|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

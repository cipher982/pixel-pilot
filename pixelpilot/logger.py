import logging
import os
import sys


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Get log level from environment variable or default to INFO
        log_level = os.getenv("LOGLEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level))

        # Simple stdout handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger

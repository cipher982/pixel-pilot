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

        # Console Handler using same level as logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Debug File Handler (keep detailed format for debugging)
        log_path = os.path.join(os.getenv("LOG_DIR", "/app"), "pixelpilot_debug.log")
        debug_handler = logging.FileHandler(log_path)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)

    return logger

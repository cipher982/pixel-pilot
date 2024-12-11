import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set to DEBUG to capture all log levels
        logger.setLevel(logging.DEBUG)

        # Console Handler for INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s|%(name)s|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Debug File Handler
        debug_handler = logging.FileHandler("pixelpilot_debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s|%(name)s|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)

    return logger

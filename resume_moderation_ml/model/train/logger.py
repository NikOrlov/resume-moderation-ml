import logging
import sys


def setup_logger(name, level=logging.DEBUG):
    """
    Sets up a logger with the specified name and level.

    Parameters:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter and add it to the handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(console_handler)

    return logger

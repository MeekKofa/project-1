import logging
from pathlib import Path


def setup_logging(log_file='training.log', level=logging.INFO, console_level=None):
    """Setup logging with proper format and handlers

    Args:
        log_file: Path to log file (default: 'training.log')
        level: Logging level for both file and console handlers (default: logging.INFO)
        console_level: Optional separate logging level for console handler (default: same as level)

    Returns:
        Root logger instance
    """
    if console_level is None:
        console_level = level
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, console_level))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

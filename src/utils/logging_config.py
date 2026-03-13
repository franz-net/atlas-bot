"""
Logging configuration utilities for Project Atlas.

Provides file-only logging with rolling timestamps and symlink management.
Inherited from Midas.
"""

import logging
import os
from pathlib import Path


def create_or_update_log_symlink(log_file_path: Path, prefix: str) -> None:
    """
    Create or update a symlink pointing to the latest log file.

    Creates a symlink like 'logs/tradestation_client_latest.log' that always
    points to the most recent timestamped log file.

    Args:
        log_file_path: Path to the current timestamped log file
        prefix: Log file prefix (e.g., 'tradestation_client')
    """
    try:
        log_dir = log_file_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = log_dir / f"{prefix}_latest.log"

        # Remove existing symlink if present
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()

        # Create new symlink pointing to the latest log file
        symlink_path.symlink_to(log_file_path.resolve())

    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not create log symlink: {e}")


def setup_file_logger(name: str, log_prefix: str) -> logging.Logger:
    """
    Set up a file-only logger with no stdout propagation.

    Args:
        name: Logger name (typically __name__)
        log_prefix: Prefix for log file (e.g., 'scheduler')

    Returns:
        Configured logger instance
    """
    from datetime import datetime

    logger = logging.getLogger(name)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamped file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{log_prefix}_{timestamp}.log'

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG if log_level == 'DEBUG' else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create symlink to latest
    create_or_update_log_symlink(log_file, log_prefix)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

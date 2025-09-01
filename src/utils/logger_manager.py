import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerManager:
    """
    A reusable logging system for Python projects.

    Features:
    - Console + File logging
    - Rotating file handler (prevents large log files)
    - Configurable log level
    - Standardized format with timestamp, module, and log level
    """

    def __init__(
        self,
        name: str = "app",
        log_dir: str = "logs",
        log_file: str = "app.log",
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 5,
    ):
        """
        Initialize the logging system.

        Args:
            name (str): Logger name, usually __name__.
            log_dir (str): Directory for log files.
            log_file (str): File name for logs.
            level (int): Logging level (DEBUG, INFO, etc.).
            max_bytes (int): Max size per log file before rotation.
            backup_count (int): Number of rotated files to keep.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs

        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Logging format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File Handler (with rotation)
        file_handler = RotatingFileHandler(
            Path(log_dir) / log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger."""
        return self.logger
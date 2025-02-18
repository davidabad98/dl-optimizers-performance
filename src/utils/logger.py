import logging
import os
import sys
from datetime import datetime


class Logger:
    """A custom logger to capture print statements and save them to a log file."""

    def __init__(self, log_dir="results", log_filename_prefix="training_log"):
        """Initialize the logger and redirect stdout and stderr."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_filename_prefix}_{timestamp}.txt"  # Append timestamp
        self.log_path = os.path.join(log_dir, log_filename)
        self._setup_logging(log_dir)

        # Redirect stdout and stderr to this logger
        sys.stdout = self
        sys.stderr = self

    def _setup_logging(self, log_dir):
        """Setup logging configurations."""
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def write(self, message):
        """Write messages to both console and log file."""
        sys.__stdout__.write(message)  # Print to console
        logging.info(message.strip())  # Log message

    def flush(self):
        """Flush output (needed when redirecting stdout)."""
        sys.__stdout__.flush()

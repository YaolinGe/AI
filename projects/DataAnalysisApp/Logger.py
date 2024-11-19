"""
Logger to keep track of the detailed information during the process.

Created on 2024-11-14
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""

import logging
import os
from datetime import datetime


class Logger:
    """
    A robust and scalable logger class that provides a single instance for use across multiple classes.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """
        Initialize the logger with appropriate settings.
        """
        # Create log directory if it doesn't exist
        log_dir = ".logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
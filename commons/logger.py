import logging

# ANSI escape sequences for colors
COLOR_RESET = "\033[0m"
COLOR_MAP = {
    "DEBUG": "\033[94m",     # Blue
    "INFO": "\033[92m",      # Green
    "WARNING": "\033[93m",   # Yellow/Orange
    "ERROR": "\033[91m",     # Red
    "CRITICAL": "\033[91m",  # Red
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_MAP.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{COLOR_RESET}"

# Create and configure the logger
logger = logging.getLogger("TMR")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    base_format = "[%(levelname)s] %(asctime)s - %(message)s"
    formatter = ColorFormatter(base_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

import logging
import sys

def setup_logging(app_name: str) -> logging.Logger:
    """Configure structured logging for the application."""
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Avoid duplicate logs if handlers already exist
    if not logger.handlers:
        logger.addHandler(console_handler)
        
    return logger

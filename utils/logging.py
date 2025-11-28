"""Centralized logging configuration for the application."""
import logging
import sys
from tqdm import tqdm

class MinimalFormatter(logging.Formatter):
    """Minimal formatter that only shows the message for INFO level."""
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"

def setup_logging(level=logging.WARNING):
    """Set up logging configuration for the application.
    
    Args:
        level: Logging level to use (default: WARNING)
    """
    # Configure root logger
    logging.basicConfig(level=level)
    
    # Suppress external loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    # logging.getLogger("funnels").setLevel(logging.WARNING)  # Commented out for debugging
    logging.getLogger("RAG").setLevel(logging.WARNING)
    
    # Enable detailed logging for our modules
    logging.getLogger("utils.document_processing.processor").setLevel(level)
    logging.getLogger("funnels.llm_provider").setLevel(level)
    
    # Set up application logger with minimal formatting
    logger = logging.getLogger("numaira")
    logger.setLevel(logging.INFO)
    
    # Create console handler with minimal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(MinimalFormatter())
    logger.addHandler(console_handler)
    
    # Configure tqdm to be less verbose
    tqdm.monitor_interval = 0
    
    return logger

# logger.py
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG if you want more details
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),   # Save logs to file
        logging.StreamHandler()          # Print logs to console too
    ]
)

# Get a root logger (you can import this anywhere)
logger = logging.getLogger("Investimate")

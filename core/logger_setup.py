import logging
from datetime import datetime
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create a single log file per session
log_filename = datetime.now().strftime("logs/matrix_log_%Y%m%d_%H%M%S.log")
logger = logging.getLogger("MatrixLogger")
logger.setLevel(logging.INFO)

# Only add handlers once
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

import sys
import os
from loguru import logger
from rag_app.config.settings import settings

# Remove default handler
logger.remove()

# Get log level from settings
log_level = settings.log_level.upper()

# Console handler for development
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True
)

# File handler for production logging
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    compression="gz",
    level=log_level,
    serialize=True,  # JSON format for structured logging
    backtrace=True,
    diagnose=True
)

# Optional: Add a separate error log
logger.add(
    "logs/error.log",
    rotation="1 day",
    retention="30 days",
    compression="gz",
    level="ERROR",
    serialize=True,
    backtrace=True,
    diagnose=True
)
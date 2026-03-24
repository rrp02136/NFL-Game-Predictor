"""
Input/Output Utility Functions for the NFL Prediction Project
Handles file operations, model persistence, and logging setup
"""
import os
import logging
import joblib
import pandas as pd
from pathlib import Path

def ensure_directory(path: str) -> None:
    # Create directory if doesn't exist, including parent directories
    Path(path).mkdir(parents=True, exist_ok=True)
    logging.debug(f"Ensured directory exists: {path}")

def save_model(model, filepath: str) -> None:
    # Save trained model using joblib
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved successfully to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save model to {filepath}: {str(e)}")
        raise

def load_model(filepath: str):
    # Load trained model using joblib
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        logging.info(f"Model loaded successfully from: {filepath}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {filepath}: {str(e)}")
        raise

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    # Save DataFrame to CSV
    try:
        ensure_directory(os.path.dirname(filepath))
        df.to_csv(filepath, index=index)
        logging.info(f"DataFrame saved successfully to: {filepath} (shape: {df.shape})")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {filepath}: {str(e)}")
        raise

def load_dataframe(filepath: str) -> pd.DataFrame:
    # Load DataFrame from CSV
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"DataFrame loaded successfully from: {filepath} (shape: {df.shape})")
        return df
    except Exception as e:
        logging.error(f"Failed to load DataFrame from {filepath}: {str(e)}")
        raise

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    # Configure Python logging with console and optional file output
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Set up handlers
    handlers = [logging.StreamHandler()]  # Console handler

    if log_file:
        ensure_directory(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))  # File handler

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True
        )

        logging.info("Logging initialized")
        if log_file:
            logging.info(f"Log file: {log_file}")
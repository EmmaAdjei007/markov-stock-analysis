import logging

def setup_logging():
    """
    Configure logging for the project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("reports/logs.log"),
            logging.StreamHandler()
        ]
    )
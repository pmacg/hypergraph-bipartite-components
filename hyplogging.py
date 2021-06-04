"""
This file configures the logging for this project.
"""
import logging
import sys

# Create the logging object.
logger = logging.getLogger('hyplap')

# Set the logging level - you may want to change this to logging.INFO for less verbose logging.
logger.setLevel(logging.DEBUG)

# Create a logging handler with a format
handler = logging.FileHandler('hyplap.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s')
handler.setFormatter(formatter)

# ... and set the handler
logger.addHandler(handler)

# Add another handler to print INFO and above to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_formatter = logging.Formatter("%(message)s")
stdout_handler.setFormatter(stdout_formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

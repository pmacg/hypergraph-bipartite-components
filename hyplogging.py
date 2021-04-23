import logging

# Create the logging object.
logger = logging.getLogger('hyplap')

# Set the logging level - you may want to change this to logging.INFO for less verbose logging.
logger.setLevel(logging.INFO)

# Create a logging handler with a format
handler = logging.FileHandler('hyplap.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s')
handler.setFormatter(formatter)

# ... and set the handler
logger.addHandler(handler)

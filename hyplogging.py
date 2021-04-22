import logging

logger = logging.getLogger('hyplap')
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('hyplap.log')

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

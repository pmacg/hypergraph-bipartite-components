import logging

logger = logging.getLogger('hyplap')
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('hyplap.log')

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

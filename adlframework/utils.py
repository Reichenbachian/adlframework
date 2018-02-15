from logging.handlers import TimedRotatingFileHandler
import logging

logger = logging.getLogger("Logger_1")
LOG_NAME = 'adlframework.log'

def get_logger():
	'''
	Makes only one logger for the whole project.
	'''
	global logger
	handler = TimedRotatingFileHandler(LOG_NAME)
	logger.addHandler(handler)
	return logger


def in_ipynb():
	'''
	Checks whether the code is currently being executed in a python notebook.
	'''
	try:
		cfg = get_ipython().config
	except NameError:
		return False
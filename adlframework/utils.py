from logging.handlers import TimedRotatingFileHandler
import logging

logger = logging.getLogger("Logger_1")

def get_logger():
	'''
	Makes only one logger for the whole project.
	'''
	global logger
	handler = TimedRotatingFileHandler('log')
	logger.addHandler(handler)
	return logger


def in_ipynb():
	'''
	Checks whether the code is currently being executed in a python notebook.
	'''
	try:
		cfg = get_ipython().config 
		if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
			return True
		else:
			return False
	except NameError:
		return False
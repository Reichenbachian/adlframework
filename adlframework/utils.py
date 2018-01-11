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
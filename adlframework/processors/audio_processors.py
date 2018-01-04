
def wav_normalize(sample):
	'''
	Normalizes a numpy array wave by dividing by
	the maximum a wave array would have.
	Max val = 2**15 = 32768 
	'''
	data, label = sample
	return data/32768, label
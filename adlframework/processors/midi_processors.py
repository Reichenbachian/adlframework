import pdb
import numpy as np
from madmom.utils import suppress_warnings
########################
#### PROCESSORS   ######
########################

@suppress_warnings # believe me. You don't want thousands of errors.
def midi_to_np(sample, targets=[], reverse=None):
	'''
	Converts a music21 midi stream object to a numpy array.
	'''
	stream, label = sample
	return stream.notes(), label


###############################
#### MADMOM PROCESSORS   ######
###############################

chro = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
chromatic = [chro[x%len(chro)]+str(x//len(chro)) for x in range(88)]
i_to_note = {0: "Whole",1: "Half",2: "Quarter",3: "Eighth",4: "Sixteenth",5: "Third",6: "Sixth",7: "Seventh",8: "Dotted Whole",9: "Dotted Half",10: "Dotted Quarter",11: "Chord"}
possible_durations = np.array([4, 2, 1, .5, .25, 1/3.0, 2/3.0, 4/7.0, 6, 3, 1.25])
possible_onsets = possible_durations + [0]
def notes_to_classification(sample):
	'''
	Converts continuous onset and duration times to the nearest note value.
	Also converts notes to chromatic names.
	Designed for madmom `.notes()` array.
	Use for classification.
	'''
	num_notes = 88
	num_note_types = 11

	data, label = sample
	#### Once for data
	duration = [np.argmin(np.abs(possible_durations - x[2])) for x in data]
	onsets = [np.argmin(np.abs(possible_onsets - x[0])) for x in data]
	notes = [min(int(x[1]), 87) for x in data]
	data = np.stack([duration, onsets, notes], axis=1)
	return data, label

def convert_to_matrix(sample):
	'''
	Converts discrete output into a matrix
	'''
	raise NotImplemented('This is not implemented. This should be implemented in the general case')
	# data = np.zeros((len(data), num_notes, num_note_types+1, num_note_types))
	# for i in range(len(notes)):
	# 	data[i][notes[i]][onsets[i]][types[i]] = 1


def make_time_relative(sample):
	'''
	Makes the first column relative rather than absolute time in
	a madmom `.notes()` numpy array.
	'''
	data, label = sample
	for i in range(len(data)-1, 0, -1):
		data[i][0] = data[i][0] - data[i-1][0]# Assume timestamps are at 0
	data[0][0] = 0
	return data, label
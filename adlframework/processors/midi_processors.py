import pdb
import numpy as np
########################
#### PROCESSORS   ######
########################

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
possible_notes = np.array([4, 2, 1, .5, .25, .33, .66, .5714, 6, 3, 1.25])
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
	types = [np.argmin(np.abs(possible_notes - x[2])) for x in data]
	onsets = [np.argmin(np.abs(possible_notes - x[0])) for x in data]
	notes = [min(int(x[1]), 87) for x in data]
	data = np.zeros((len(data), num_notes, num_note_types+1, num_note_types))
	for i in range(len(notes)):
		data[i][notes[i]][onsets[i]][types[i]] = 1

	#### Once for label
	types = [np.argmin(np.abs(possible_notes - x[2])) for x in label]
	onsets = [np.argmin(np.abs(possible_notes - x[0])) for x in label]
	notes = [int(min(x[1], 87)) for x in label]
	label = np.zeros((len(label), num_notes, num_note_types+1, num_note_types))
	for i in range(len(notes)):
		label[i][notes[i]][onsets[i]][types[i]] = 1

	return data, label


def make_time_relative(sample):
	'''
	Makes the first column relative rather than absolute time in
	a madmom `.notes()` numpy array.
	'''
	data, label = sample
	for tmp in [data, label]:
		for i in range(len(tmp)-1, 0, -1):
			tmp[i][0] = tmp[i][0] - tmp[i-1][0]# Assume timestamps are at 0
		tmp[0][0] = 0
	return data, label
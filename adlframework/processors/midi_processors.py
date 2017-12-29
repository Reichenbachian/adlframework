########################
#### PROCESSORS   ######
########################

def midi_to_np(sample, targets=[], reverse=None):
    '''
    Converts a music21 midi stream object to a numpy array.
    '''
    stream, label = sample
    return stream.notes(), label

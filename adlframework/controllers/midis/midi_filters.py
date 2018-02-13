import pdb
########################
######  Filters  #######
########################

def num_instruments(sample, n=None):
    '''
    Limit the number of instruments
    '''
    assert n != None, 'Please give a non-None value to n in num_instruments.'
    stream, label = sample
    pdb.set_trace()
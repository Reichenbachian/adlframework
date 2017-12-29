import pdb
def crop_and_label(sample, num_rows):
    '''
    Takes the last `n` rows and converts them into a label
    while cropping the original sample.
    '''
    arr, label = sample
    split = len(arr)-num_rows
    label = arr[split:]
    arr = arr[:split]
    return arr, label
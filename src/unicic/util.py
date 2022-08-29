#!/usr/bin/env python3
'''
Utility functions which are generic to the array library
'''

def rebatch(array, size, axis=0):
    '''Partition array to tuple of rebatched and remainders.

    The rebatched array will have given axis become the given size and
    then gain an additional first axis which size that of the number
    of batches.

    The remainder array will have same shape as the input array but
    with given axis with size less than given size

    Eg an (11,3) array rebatched with size 5 will return a rebatched
    array of shape(2,5,3) and a remainder shaped (1,3)

    If the array may be perfectly rebatched the remainder array will
    be shaped with a zero size on given axis.

    '''
    ndim = len(array.shape)
    
    nbatches = array.shape[axis] // size
    end = nbatches * size

    rems = [slice(None)]*ndim
    rems[axis] = slice(end, None)
    remainder = array[tuple(rems)]

    rebs = [slice(0,s) for s in array.shape]
    rebs[axis] = slice(0,end)
    rebatched = array[tuple(rebs)]

    res = list(rebatched.shape)

    nbatches = res[axis] // size
    res[axis] = size
    res.insert(0, nbatches)
    rebatched = rebatched.reshape(res)    

    return rebatched, remainder



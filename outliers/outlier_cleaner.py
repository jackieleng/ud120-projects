#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np
    errors = (predictions - net_worths)**2

    a = np.hstack((ages, net_worths, errors))
    # sort the third column
    aa = a[a[:, 2].argsort()]
    # take 90% lowest errors
    ninety = int(0.9 * aa.shape[0])
    aa = aa[:ninety]
    cleaned_data = map(tuple, aa)
    #import pdb; pdb.set_trace()

    
    return cleaned_data


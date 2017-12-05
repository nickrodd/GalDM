###############################################################################
# prop_mod.py
###############################################################################
#
# Calculate mod without numpy issue
#
###############################################################################

import numpy as np
from tqdm import *

def mod(dividends, divisor):
    """ return dividends (array) mod divisor (double)
    """

    output = np.zeros(len(dividends))

    for i in tqdm(range(len(dividends))): 
        output[i] = dividends[i]
        done=False
        while (not done):
            if output[i] >= divisor:
                output[i] -= divisor
            elif output[i] < 0.:
                output[i] += divisor
            else:
                done=True

    return output

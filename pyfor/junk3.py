import numpy as np

def perform( fun, *args ):
    fun( *args )

def mean(values):
    print(np.mean(values))

def action2( args ):
    print("action2")

perform( mean, [1,2,3] )

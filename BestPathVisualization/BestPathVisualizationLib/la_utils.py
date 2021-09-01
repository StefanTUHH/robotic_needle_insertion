import numpy as np

def normalized(v):
    return v/np.sqrt( np.dot(v,v))


def ortho_proj_vec(v, uhat ):
    '''Returns the projection of the vector v  on the subspace
    orthogonal to uhat (which must be a unit vector) by subtracting off
    the appropriate multiple of uhat.
    i.e. dot( retval, uhat )==0
    '''
    return v-np.dot(v,uhat)*uhat


def ortho_proj_array( Varray, uhat ):
    ''' Compute the orhogonal projection for an entire array of vectors.
    @arg Varray:  is an array of vectors, each row is one vector
        (i.e. Varray.shape[1]==len(uhat)).
    @arg uhat: a unit vector
    @retval : an array (same shape as Varray), where each vector
            has had the component parallel to uhat removed.
            postcondition: np.dot( retval[i,:], uhat) ==0.0
            for all i. 
    ''' 
    return Varray-np.outer( np.dot( Varray, uhat), uhat )
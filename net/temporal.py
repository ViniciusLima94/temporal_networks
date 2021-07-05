import numpy  as np
import xarray as xr
from   .util  import instantiate_graph, _check_inputs

def compute_temporal_correlation(A, tau = 1, mirror=False):
    # Check inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]

    if mirror: A = A + np.transpose(A, (1,0,2)) 
    
    if tau < 1:
        tau = 1

    num = (A[:,:,0:-tau] * A[:,:,tau:]).sum(axis=1)
    den = np.sqrt(A[:,:,0:-tau].sum(axis = 1) * A[:,:,tau:].sum(axis = 1))
    Ci  = np.nansum(( num / den ), axis=1) / (A.shape[-1] - 1)
    return np.nansum(Ci) / nC

def cosine_similarity(A, thr = None, mirror=False):
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]

    if mirror: A = A + np.transpose(A, (1,0,2,3)) 

    num = ( A[:,:,:,:-1]*A[:,:,:,1:] ).sum(axis = 1)
    den = np.sqrt( np.sum(A[:,:,:,:-1]**2, axis = 1) ) * np.sqrt( np.sum(A[:,:,:,1:]**2, axis = 1) )
    
    return num / den

def jaccard_index(A, mirror=False):
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of channels
    nC = A.shape[0]

    if mirror: A = A + np.transpose(A, (1,0,2,3)) 
    
    num = (A[:,:,:,:-1] * A[:,:,:,1:]).sum(axis=1)
    den = (A[:,:,:,:-1] + A[:,:,:,1:])
    #  The union is the number of elements in A plus the elements in B 
    #  minus the number of elements A and B have in common
    den[den==2] = 1
    den = den.sum(axis=1)
    J = num/den
    J[J==np.inf]=np.nan
    return J


import random
import numpy  as np 
import xarray as xr
import igraph as ig
from   tqdm   import tqdm
from   .util  import instantiate_graph, _check_inputs, _unwrap_inputs

_DEFAULT_TYPE = np.float32

def shuffle_frames(A, seed=0):

    # Checking inputs
    _check_inputs(A, 3)
    # Get values in case it is an xarray
    if isinstance(A, xr.DataArray): A = A.values

    #  Number of observations
    nt = A.shape[-1]
    #  Set randomization seed
    np.random.seed(seed)
    #  Observation indexes
    idx = np.arange(nt, dtype = int)
    np.random.shuffle(idx)
    A_null = ( A + np.transpose( A,  (1,0,2) ) ).copy()
    A_null = A_null[:,:,idx]

    return A_null.astype(_DEFAULT_TYPE)

def randomize_edges(A, n_rewires = 100, seed=0, verbose=False):
    r'''
    Randomize an adjacency matrix while maintaining its nodes' degrees
    and undirectionality.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - n_rewires: Number of rewires to be performed in each adjacency matrix.
    - seed: Seed for random edge selection.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - A_null: The randomized multiplex adjacency matrix with shape (roi,roi,trials,time).
    '''
    # Set random seed (for python's default random and numpy)
    random.seed(seed); np.random.seed(seed)

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    A_null = np.empty_like(A)

    itr = range(nt)
    for t in (tqdm(itr) if verbose else itr):
        g   = instantiate_graph(A[:,:,t], is_weighted=False)
        G   = g.copy()
        G.rewire(n=n_rewires)
        A_null[:,:,t] = np.array(list(G.get_adjacency()))

    # Unstack trials and time
    A_null = A_null.reshape( (len(roi),len(roi),len(trials),len(time)) )
    # Convert to xarray
    A_null = xr.DataArray(A_null.astype(_DEFAULT_TYPE), dims=("roi_1","roi_2","trials","time"),
                                  coords={"roi_1": roi,
                                          "roi_2": roi,
                                          "time": time, 
                                          "trials": trials} )
    return A_null

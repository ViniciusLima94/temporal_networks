import igraph as ig 
import numpy  as np
import numba  as nb
import xarray as xr
import scipy
from   scipy  import stats
from   tqdm   import tqdm

@nb.njit
def _is_binary(matrix): 
    r'''
    Check if a matrix is binary or weighted.
    > INPUT:
    - matrix: The adjacency matrix or tensor.
    '''
    is_binary = True
    for v in np.nditer(matrix):
        if v.item() != 0 and v.item() != 1:
            is_binary = False
            break
    return is_binary

def _convert_to_affiliation_vector(n_nodes, partitions):
    r'''
    Convert partitions in leidenalg format to array.
    > INPUTS:
    - n_nodes:    The number of nodes.
    - partitions: Parition objects of type "leidenalg.VertexPartition.ModularityVertexPartition"
    > OUTPUTS:
    - av: Affiliation vector.
    '''
    # Extact list of partitions
    #  partitions = partitions.values[0]
    # Number of time points
    n_times    = len(partitions)
    # Affiliation vector
    av         = np.zeros((n_nodes,n_times))
    for t in range(n_times):
        for comm_i, comm in enumerate( partitions[t] ):
            av[comm,t] = comm_i
    return av

@nb.jit(nopython=True)
def MODquality(A,av,gamma=1):
    r'''
    Given an affiliation vector compute the modularity of the graph given by A (adapted from brainconn).
    > INPUTS:
    - A: Adjacency matrix.
    - av: Affiliation vector of size (n_nodes) containing the label of the community of each node.
    - gamma: Value of gamma to use.
    > OUTPUTS:
    - The modularity index of the network.
    '''
    # Number of nodes
    n_nodes = A.shape[0]
    # Degrees
    d       = A.sum(0)
    # Number of edges
    n_edges = np.sum(d)
    # Initial modularity matrix
    B       = A - gamma * np.outer(d, d) / n_edges
    #  B       = B[av][:, av]
    # Tile affiliation vector
    s       = av.repeat(n_nodes).reshape((-1, n_nodes))#np.tile(av, (n_nodes, 1))
    return np.sum(np.logical_not(s - s.T) * B / n_edges)

#@nb.jit(nopython=True)
def CPMquality(A,av,gamma=1):
    r'''
    Constant Potts Model (CPM) quality function.
    > INPUTS:
    - A: Adjacency matrix.
    - av: Affiliation vector of size (n_nodes) containing the label of the community of each node.
    - gamma: Value of gamma to use.
    > OUTPUTS:
    - H: The quality given by the CPM model.
    '''
    av    = av.astype(int)
    # Total number of communities
    n_comm = int(np.max(av))+1
    # Quality index
    H      = 0
    for c in range(n_comm):
        # Find indexes of channels in the commucnit C
        idx = av==c
        # Number of nodes in community C
        n_c = np.sum(idx)
        H   = H + np.sum(A[np.ix_(av[idx],av[idx])])-gamma*n_c*(n_c-1)/2
        a =np.sum(A[np.ix_(av[idx],av[idx])])
        b = n_c*(n_c-1)/2
        #  print(f'{a=}')
        #  print(f'{b=}')
    return H

def _check_inputs(array, dims):
    r'''
    Check the input type and size.
    > INPUT:
    - array: The data array.
    - dims: The number of dimensions the array should have.
    '''
    assert isinstance(dims, int)
    assert isinstance(array, (np.ndarray, xr.DataArray))
    assert len(array.shape)==dims, f"The adjacency tensor should be {dims}D."

def _unwrap_inputs(array, concat_trials=False):
    r'''
    Unwrap array and its dimensions for further manipulation.
    > INPUTS:
    - array: The data array (roi,roi,trials,time).
    - concat_trials: Wheter to concatenate or not trials of the values in the array.
    > OUTPUTS:
    - array values concatenated or not and the values for each of its dimensions.
    '''
    if isinstance(array, xr.DataArray): 
        # Concatenate trials and time axis
        try:
            roi    = array.roi_1.values
            trials = array.trials.values
            time   = array.time.values
        except:
            roi    = np.arange(0, array.shape[0])
            trials = np.arange(0, array.shape[2])
            time   = np.arange(0, array.shape[3])
        if concat_trials: array = array.stack(observations=("trials","times"))
        array = array.values
    else:
        roi    = np.arange(0, array.shape[0])
        trials = np.arange(0, array.shape[2])
        time   = np.arange(0, array.shape[3])
        if concat_trials: array = array.reshape( (len(roi),len(roi),len(trials)*len(time)) )
    return array, roi, trials, time

def _reshape_list(array, shapes, dtype):
    assert isinstance(shapes, tuple)
    assert isinstance(array,  list)
    idx       = 0
    container = np.zeros(shapes, dtype=dtype)
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            container[i,j]=array[idx]
            idx += 1
    return container

def convert_to_adjacency(tensor,sources,targets):
    r'''
    Convert the tensor with the edge time-series to a matrix representations.
    > INPUTS:
    - tensor: The tensor with the edge time series (roi,freqs,trials,times).
    - sources: list of source nodes.
    - targets: list of target nodes.
    > OUTPUTS:
    - The adjacency matrix (roi,roi,freqs,trials,times).
    '''

    assert tensor.ndim==4
    assert tensor.shape[0]==len(sources)==len(targets)

    # Number of pairs
    n_pairs,n_bands,n_trials,n_times = tensor.shape[:]
    # Number of channels
    n_channels = int(np.roots([1,-1,-2*n_pairs])[0])

    # Adjacency tensor
    A = np.zeros([n_channels, n_channels, n_bands, n_trials, n_times])

    for p in range(n_pairs):
        i, j        = sources[p], targets[p]
        A[i,j,...]  = A[j,i,...] = tensor[p,...]
    return A
#  def convert_to_adjacency(tensor,):
#      # Number of pairs
#      n_pairs    = tensor.shape[0]
#      # Number of channels
#      n_channels = int(np.roots([1,-1,-2*n_pairs])[0])
#      # Number of bands
#      n_bands    = tensor.shape[1]
#      # Number of trials
#      n_trials   = tensor.shape[2]
#      # Number of time points
#      n_times    = tensor.shape[3]
#      # Channels combinations 
#      pairs      = np.transpose( np.tril_indices(n_channels, k = -1) )

#      # Adjacency tensor
#      A = np.zeros([n_channels, n_channels, n_bands, n_trials, n_times])

#      for p in range(n_pairs):
#          i, j          = int(pairs[p,0]), int(pairs[p,1])
#          A[i,j,...]  = A[j,i,...] = tensor[p,...]

#      return A

def instantiate_graph(A, is_weighted = False):
    if is_weighted:
        g = ig.Graph.Weighted_Adjacency(A.tolist(), attr="weight", loops=False, mode=ig.ADJ_UNDIRECTED)
    else:
        g = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_UNDIRECTED)
    return g

def compute_coherence_thresholds(tensor, q=0.8, relative=False, verbose=False):
    r'''
    Compute the power/coherence thresholds for the data
    > INPUTS:
    - tensor: Data with dimensions [nodes/links,bands,observations] or [nodes/links,bands,trials,time]
    - q: Quartile value to use as threshold
    - relative: If True compute one threshold for each node/link in each band (defalta False)
    > OUTPUTS:
    - coh_thr: Threshold values, if realtive is True it will have dimensions ["links","bands","trials"] otherwise ["bands","trials"] (if tensor shape is 3 there is no "trials" dimension)
    '''
    if len(tensor.shape)==4: 
        n_nodes, n_bands, n_trials, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
        if relative:
            coh_thr = np.zeros([n_nodes, n_bands, n_trials])
            itr = range(n_trials) # Iterator
            for t in (tqdm(itr) if verbose else itr):
                for i in range( n_bands ):
                    coh_thr[:,i,t] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,t,:], prob=q, axis=-1) )
            coh_thr = xr.DataArray(coh_thr, dims=("roi","freqs","trials"))
        else:
            coh_thr = np.zeros( [n_bands, n_trials] )
            itr = range(n_trials) # Iterator
            for t in (tqdm(itr) if verbose else itr):
                for i in range( n_bands ):
                    coh_thr[i,t] = stats.mstats.mquantiles(tensor[:,i,t,:].flatten(), prob=q)
            coh_thr = xr.DataArray(coh_thr, dims=("freqs","trials"))
    if len(tensor.shape)==3: 
        n_nodes, n_bands, n_obs = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        if relative:
            coh_thr = np.zeros([n_nodes, n_bands])
            itr = range(n_bands) # Iterator
            for i in (tqdm(itr) if verbose else itr):
                coh_thr[:,i] = np.squeeze( stats.mstats.mquantiles(tensor[:,i,:], prob=q, axis=-1) )
            coh_thr = xr.DataArray(coh_thr, dims=("roi","freqs"))
        else:
            coh_thr = np.zeros( n_bands )
            itr = range(n_bands) # Iterator
            for i in (tqdm(itr) if verbose else itr):
                coh_thr[i] = stats.mstats.mquantiles(tensor[:,i,:].flatten(), prob=q)
            coh_thr = xr.DataArray(coh_thr, dims=("freqs"))

    return coh_thr

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    r'''
    Check if the matrix a is symmetric
    '''
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)

import numpy            as     np
import xarray           as     xr
import igraph           as     ig
import brainconn        as     bc
import leidenalg
import warnings
from   frites.utils          import parallel_func
from   .null_models          import *
from   tqdm                  import tqdm
from   .util                 import instantiate_graph, _check_inputs, _unwrap_inputs, \
                                    _reshape_list, _is_binary, _convert_to_affiliation_vector,\
                                    MODquality, CPMquality

_DEFAULT_TYPE = np.float32

def compute_nodes_degree(A, mirror=False):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the strength (weighted) degree (binary) of each
    node is computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - mirror: If True will mirror the adjacency matrix (should be used if only the upper/lower triangle is given.
    > OUTPUTS:
    - node_degree: A matrix containing the nodes degree with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=False)

    if mirror:
        A = A + np.transpose( A, (1,0,2,3) )

    node_degree = A.sum(axis=1)

    # Convert to xarray
    node_degree = xr.DataArray(node_degree.astype(_DEFAULT_TYPE), dims=("roi","trials","times"),
                               coords={"roi": roi, "times": time, "trials": trials} )

    return node_degree

def compute_nodes_clustering(A, verbose=False, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the clustering coefficient for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - backend: Wheter to use igraph or brainconn package.
    - n_jobs: Number of jobs to use when parallelizing in observations.
    > OUTPUTS:
    - clustering: A matrix containing the nodes clustering with shape (roi,trials,time).
    '''
    assert backend in ['igraph','brainconn']
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node clustering
    clustering  = np.zeros([nC,nt])

    # Compute for a single observation
    def _for_frame(t):
        #  Instantiate graph
        if is_weighted:
            if backend == 'igraph':
                g          = instantiate_graph(A[...,t], is_weighted=is_weighted)
                clustering = g.transitivity_local_undirected(weights="weight")
            elif backend == 'brainconn':
                clustering = bc.clustering.clustering_coef_wu(A[...,t])
        else:
            if backend == 'igraph':
                g          = instantiate_graph(A[...,t], is_weighted=is_weighted)
                clustering = g.transitivity_local_undirected()
            elif backend == 'brainconn':
                clustering = bc.clustering.clustering_coef_bu(A[...,t])
        return clustering

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    clustering = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    clustering = np.asarray(clustering).T

    # Unstack trials and time
    clustering = clustering.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    clustering = xr.DataArray(np.nan_to_num(clustering).astype(_DEFAULT_TYPE), dims=("roi","trials","times"),
                              coords={"roi":roi, "times":time, "trials":trials} )

    return clustering

def compute_nodes_coreness(A, verbose=False, n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the coreness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - n_jobs: Number of jobs to use when parallelizing in observations.
    > OUTPUTS:
    - coreness: A matrix containing the nodes coreness with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node coreness
    #  coreness  = np.zeros([nC,nt])

    # Compute for a single observation
    def _for_frame(t):
        g        = instantiate_graph(A[...,t], is_weighted=is_weighted)
        coreness = g.coreness()
        return coreness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    coreness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    coreness = np.asarray(coreness).T

    # Unstack trials and time
    coreness = coreness.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    coreness = xr.DataArray(coreness.astype(_DEFAULT_TYPE), dims=("roi","trials","times"),
                              coords={"roi": roi, "times": time, "trials": trials} )

    return coreness

def compute_nodes_coreness_bc(A, verbose=False, n_jobs=1):
    r'''
    The same as 'compute_nodes_coreness' but based on brainconnectivity toolbox method can be either for binary
    or weighted undirected graphs.
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the coreness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - n_jobs: Number of jobs to use when parallelizing in observations.
    > OUTPUTS:
    - coreness: A matrix containing the nodes coreness with shape (roi,trials,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]
    #  Variable to store node coreness

    ##################################################################
    # Include method for S-core
    #################################################################
    def _nodes_kcore_bu(A):
        # Number of nodes
        n_nodes = len(A)
        # Initial coreness
        k       = 0
        # Store each node's coreness
        k_core  = np.zeros(n_nodes)
        # Iterate until get a disconnected graph
        while True:
            # Get coreness matrix and level of k-core
            C, kn = bc.core.kcore_bu(A,k,peel=False)
            if kn==0:
                break
            # Assigns coreness level to nodes
            idx = C.sum(1)>0
            k_core[idx]=k
            k+=1
        return k_core+1

    # Compute for a single observation
    def _for_frame(t):
        coreness = _nodes_kcore_bu(A[...,t])
        return coreness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    coreness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    coreness = np.asarray(coreness).T

    # Unstack trials and time
    coreness = coreness.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    coreness = xr.DataArray(coreness.astype(_DEFAULT_TYPE), dims=("roi","trials","times"),
                              coords={"roi": roi, "times": time, "trials": trials} )

    return coreness

def compute_nodes_betweenness(A, verbose=False, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the betweenness for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - n_jobs: Number of jobs to use when parallelizing in observations.
    > OUTPUTS:
    - betweenness: A matrix containing the nodes betweenness with shape (roi,time).
    '''
    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    # Compute for a single observation
    def _for_frame(t):
        if is_weighted:
            if backend == 'igraph':
                g           = instantiate_graph(A[...,t], is_weighted=is_weighted)
                betweenness = g.betweenness(weights="weight")
            elif backend == 'brainconn':
                betweenness = bc.centrality.betweenness_wei(A[...,t])
        else:
            if backend == 'igraph':
                g           = instantiate_graph(A[...,t], is_weighted=is_weighted)
                betweenness = g.betweenness()
            elif backend == 'brainconn':
                betweenness = bc.centrality.betweenness_bin(A[...,t])
        return betweenness

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    betweenness = parallel(p_fun(t) for t in range(nt))
    # Convert to numpy array
    betweenness = np.asarray(betweenness).T

    # Unstack trials and time
    betweenness = betweenness.reshape( (len(roi),len(trials),len(time)) )
    # Convert to xarray
    betweenness = xr.DataArray(betweenness.astype(_DEFAULT_TYPE), dims=("roi","trials","times"),
                              coords={"roi": roi, "times": time, "trials": trials} )

    return betweenness

def compute_network_partition(A,  kw_louvain={}, kw_leiden={}, verbose=False, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the network partition for each
    node is computed for all the trials concatenated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_leiden: Parameters to be passed to leindelalg (for further info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    > OUTPUTS:
    - partition: A list with the all the partition found for each layer of the
    matrix (for each observation or trials,time if flatten is False).
    '''

    assert backend in ['igraph','brainconn']

    # Check inputs
    _check_inputs(A, 4)
    # Get values in case it is an xarray
    A, roi, trials, time = _unwrap_inputs(A,concat_trials=True)
    # Check if the matrix is weighted or binary
    is_weighted = not _is_binary(A)
    #  Number of channels
    nC = A.shape[0]
    #  Number of observations
    nt = A.shape[-1]

    # Using igraph
    if backend == 'igraph':
        #  Save the partitions
        # Store nodes' membership
        partition = np.zeros((nC,nt))
        # Store network modularity
        modularity  = np.zeros(nt)

        itr = range(nt)
        for t in (tqdm(itr) if verbose else itr):
            g          = instantiate_graph(A[...,t], is_weighted=is_weighted)
            # Uses leidenalg
            if is_weighted:
                weights="weight"
            else:
                weights=None

            optimizer = leidenalg.ModularityVertexPartition
            # Find partitions
            partition[:,t]= leidenalg.find_partition(g, optimizer, weights=weights, **kw_leiden).membership
            # Compute modularity
            modularity[t] = MODquality(A[...,t],partition[:,t],1)

    # Using brainconn
    elif backend == 'brainconn':

        def _for_frame(t):
            partition, modularity  = bc.modularity.modularity_louvain_und(A[...,t], **kw_louvain)
            return np.concatenate((partition-1,[modularity]))
            #  return partition-1, modularity

        # define the function to compute in parallel
        parallel, p_fun = parallel_func(
            _for_frame, n_jobs=n_jobs, verbose=verbose,
            total=nt)

        # Compute the single trial coherence
        #  partition, modularity = parallel(p_fun(t) for t in range(nt))
        out = np.squeeze( parallel(p_fun(t) for t in range(nt)) )
        partition, modularity = np.asarray(out[:,:-1]).T, np.asarray(out[:,-1])

    # Reshape partition and modularity back to trials and time
    partition = np.reshape(partition, (nC,len(trials),len(time)))
    # Conversion to xarray
    partition = xr.DataArray(partition.astype(int), dims=("roi","trials","times"),
                             coords={"roi":roi,"trials":trials,"times":time})

    # Unstack trials and time 
    modularity = modularity.reshape( (len(trials),len(time)) )
    # Convert to xarray
    modularity = xr.DataArray(modularity.astype(_DEFAULT_TYPE), dims=("trials","times"),
                              coords={"times": time, "trials": trials} )

    return partition, modularity

def compute_allegiance_matrix(A, kw_louvain={}, kw_leiden={}, concat=False, verbose=False, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials*time), the allegiance matrix for
    the whole period provided will be computed.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_louvain: Parameters to be passed to louvain alg for brainconn package (for further info see: brainconn.readthedocs.io/en/latest/generated/brainconn.modularity.community_louvain.html#brainconn.modularity.community_louvain)
    - kw_leiden: Parameters to be passed to leindelalg (for frther info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
    - concat: Wheter trials are concatenated or not.
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - backend: Wheter to use igraph or brainconn package.
    - n_jobs: Number of jobs to use when parallelizing in observations.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi)
    '''

    assert backend in ['igraph','brainconn']

    # Number of ROI
    nC           = A.shape[0]
    # Getting roi names
    if isinstance(A, xr.DataArray):
        roi = A.sources.values
    else:
        roi = np.arange(nC, dtype=int)

    # Using igraph
    if backend == 'igraph':
        assert n_jobs==1, "For backend igraph n_jobs is not allowed" #  
        #  Find the partitions 
        if verbose: print("Finding network partitions.\n")
        p,_ = compute_network_partition(A, kw_leiden, verbose=verbose,backend='igraph')
    # Using brainconn
    elif backend == 'brainconn':
        #  Find the partitions
        if verbose: print("Finding network partitions.\n")
        p,_ = compute_network_partition(A, kw_louvain, verbose=verbose,backend='brainconn',n_jobs=n_jobs)

    # Getting dimension arrays
    trials, time = p.trials.values, p.times.values
    # Total number of observations
    nt           = len(trials)*len(time)
    # Stack paritions 
    p            = p.stack(observations=("trials","times"))

    def _for_frame(t):
        # Allegiance for a frame
        T  = np.zeros((nC,nC))
        # Affiliation vector
        av = p.isel(observations=t).values
        # For now convert affiliation vector to igraph format
        n_comm = int(av.max()+1)
        for j in range(n_comm):
            p_lst = np.arange(nC,dtype=int)[av==j]
            grid  = np.meshgrid(p_lst,p_lst)
            grid  = np.reshape(grid, (2, len(p_lst)**2)).T
            T[grid[:,0],grid[:,1]] = 1
        np.fill_diagonal(T,1)
        return T

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_frame, n_jobs=n_jobs, verbose=verbose,
        total=nt)
    # Compute the single trial coherence
    T = parallel(p_fun(t) for t in range(nt))
    T = np.nanmean(T,0)

    # Converting to xarray
    T = xr.DataArray(T.astype(_DEFAULT_TYPE), dims=("sources","targets"),
                     coords={"sources":roi, "targets": roi})
    return T

def windowed_allegiance_matrix(A, kw_louvain={}, kw_leiden={}, times=None,  verbose=False, win_args=None, backend='igraph', n_jobs=1):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), the windowed allegiance matrix.
    For each window the observations are concatenated for all trials and then the allegiance matrix is estimated.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - kw_louvain: Parameters to be passed to louvain alg for brainconn package (for frther info see: brainconn.readthedocs.io/en/latest/generated/brainconn.modularity.community_louvain.html#brainconn.modularity.community_louvain)
    - kw_leiden: Parameters to be passed to leindelalg (for frther info see: https://leidenalg.readthedocs.io/en/stable/reference.html)
    - times: Time array to construct the windows.
    - is_weighted: Scepecify if the network is weighted or binary.
    - verbose: Wheater to print the progress or not.
    - win_args: Dict. with arguments to be passed to define_windows (for more details see frites.conn.conn_sliding_windows)
    - backend: Wheter to use igraph or brainconn package.
    - n_jobs: Number of jobs to use when parallelizing over windows.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi, trials, time)
    '''
    from frites.conn.conn_sliding_windows import define_windows

    assert isinstance(win_args, dict)
    assert isinstance(A, xr.DataArray)
    assert ('times' in A.dims) and ('trials' in A.dims) and ('sources' in A.dims) and ('targets' in A.dims)

    # Number of regions
    nC         = A.shape[0]
    # ROIs
    roi        = A.sources.values
    # Define windows
    win, t_win = define_windows(times, **win_args)
    # For a given trial computes windowed allegiance
    def _for_win(trial, win):
        T = xr.DataArray(np.zeros((nC,nC,len(win))),
                         dims=("sources","targets","times"),
                         coords={"sources":roi, "targets": roi, "times":t_win})
        for i_w, w in enumerate(win):
            T[...,i_w]=compute_allegiance_matrix(A.isel(trials=[trial],times=slice(w[0],w[1])),
                                                 kw_louvain, kw_leiden, verbose=verbose,
                                                 backend=backend, n_jobs=1)
        return T.astype(_DEFAULT_TYPE)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_win, n_jobs=n_jobs, verbose=verbose,
        total=A.shape[2])
    # compute the single trial coherence
    T = parallel(p_fun(trial,win) for trial in range(A.shape[2]))
    # Concatenating
    T = xr.concat(T, dim="trials")
    # Ordering dimensions
    T = T.transpose("sources","targets","trials","times")
    # Assign time axis
    T = T.assign_coords({"trials":A.trials.values})
    return T

def null_model_statistics(A, f_name, n_stat, n_rewires=1000, seed=0, n_jobs=1,  **kwargs):
    r'''
    Given the multiplex adjacency matrix A with shape (roi,roi,trials,time), compute the null-statistic
    of a given measurement for different repetitions/seeds.
    > INPUTS:
    - A: Multiplex adjacency matrix with shape (roi,roi,trials,time).
    - f_name: The name of the function of wich the null-statistic should be computed.
    - n_stat: The number of different random seeds to use to compute the null-statistic.
    - seed: Initial seed to set other seeds.
    - n_rewires: The number of rewires to be applied to the binary adjacency matrix,
    - n_jobs: Number of jobs to use when parallelizing over windows.
    > OUTPUTS:
    - T: The allegiance matrix between all nodes with shape (roi, roi, trials, time)
    '''

    assert f_name.__name__ in ['compute_nodes_degree','compute_nodes_clustering','compute_nodes_coreness','compute_nodes_betweenness','compute_network_modularity']

    # Compute the null statistics for a given seed
    def _single_estimative(A, f_name, n_rewires, seed, **kwargs):
        #  Create randomized model
        A_null = randomize_edges(A,n_rewires=n_rewires,seed=seed)
        return f_name(A_null, **kwargs)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _single_estimative, n_jobs=n_jobs, verbose=False,
        total=n_stat)
    # compute the single trial coherence
    measures = parallel(p_fun(A,f_name,n_rewires,i*(seed+100),**kwargs) for i in range(n_stat))

    # Converting to xarray
    measures = xr.concat(measures, dim='seeds')

    return measures

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def get_neighborhoods_with_null(mat, ct, center, cell, k, niter):
    '''
    Gets the local neighborhood expression, as well as a null distribution
    for cell type "cell"
    ---------------------
    parameters:
        mat: digital gene expression matrix
        ct: vector of all cell types
        center: x, y, z coordinates for each cell
        cell: cell type to find spDEGs for
        k: Number of cells in a local neighborhood
        niter: number of permutations
    ---------------------
    returns:
        gene_mat: local neighborhood expression
        nullm: Null distribution of local neighborhood expression
    '''
    # Finds where the cells of a given cell type are
    locs = np.where(ct == cell)[0]
    sub_mat = mat.iloc[locs, :]

    # hard coded, if number of cells is less than
    # 4 times the neighborhood size, don't compute spDEGs
    # this can be changed
    if sub_mat.shape[0] < 4 * k:
        print('Only ', str(sub_mat.shape[0]), ' cells present')
        return None

    # Train KNN classifier to find neighborshoods
    sub_cent = center[locs, :]
    sub_ct = ct[locs]
    clf = KNeighborsClassifier(n_neighbors=k).fit(sub_cent, sub_ct)
    dist, ids = clf.kneighbors(sub_cent)

    # get the null distribution of neighobrhoods
    nullm = get_null(sub_mat, ids, niter)

    # get the real neighborhood expression
    gene_mat = get_local_neigh(sub_mat, ids)

    return gene_mat, nullm


def get_null(sm, ids, niter):
    '''
    Gets the null distribution of local neighborhoods
    ---------------------
    parameters:
        sm: DGE of cells in the current cell type
        ids: indices of nearest neighbors
        niter: Number of permutations
    ---------------------
    returns:
        nullmat: null distribution of local neighborhood expression
    '''
    nullmat = np.zeros((niter,
                        sm.shape[0],
                        sm.shape[1]))
    ids_rand = ids.copy()
    for i in range(niter):
        np.random.shuffle(ids_rand.ravel())
        nullmat[i, :, :] = get_local_neigh(sm, ids_rand)
    return nullmat


def get_local_neigh(cm, ids):
    '''
    Gets the local expression of a neighborhood around each cell
    ---------------------
    parameters:
        cm: DGE of cells in current cell type
        ids: indices of nearest neighbors
    ---------------------
    returns:
        neigh_mat: local neighborhood expression around each cell
    '''
    neigh_mat = np.zeros_like(cm)
    for i in range(ids.shape[0]):
        temp_mat = cm.iloc[ids[i, :], :]
        neigh_mat[i, :] = np.mean(temp_mat, axis=0)

    return neigh_mat

def get_spatial_pval(cells_mat, celltypes, cell_cent, ct, nneighbors, nperm):
    '''
    Get the pvalue of spDEGs for each gene in a given cell type
    ---------------------
    parameters:
        cells_mat: DGE matrix (cells x genes)
        celltypes: vector of cell types
        cell_cent: locations of each cell, (cells x euclidean (xyz))
        ct: cell type of interest
        nneighbors: number of neighbors in a local neighborhood
        nperm: number of permutation to generate a null distribution
    ---------------------
    returns:
        ps_mat_raveled: list of p values, and gene indices
        returns: None if there aren't enough cells
    '''
    neighborhoods_output = get_neighborhoods_with_null(cells_mat, celltypes,
                                      cell_cent, ct,nneighbors,
                                      nperm)

    if neighborhoods_output is not None:

        gm, nm = neighborhoods_output

        ps_mat_raveled = []
        for i in range(gm.shape[1]):
            nm_rav = nm[:, :, i]
            if np.max(gm[:, i]) > 5:
                var_vec = np.var(nm_rav, axis=1)
                real_var = np.var(gm[:, i])

                p = 1 - (np.sum(real_var > var_vec) / len(var_vec))

                ps_mat_raveled.append([p, i])

        return ps_mat_raveled

    else:
        print('Not enough cells in cell type.')
        return None

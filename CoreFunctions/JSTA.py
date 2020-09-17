import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from math import exp, sqrt, log
import scipy.spatial
import os
from time import time
from collections import Counter
from sklearn.preprocessing import scale
from skimage.segmentation import watershed
from numpy.ctypeslib import ndpointer
import ctypes
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import neighbors

path_to_file = "/home/russell/hoffman/JSTA/CoreFunctions"
#REPLACE-WITH-PATH

def combine_matrices_and_predict_probs(ref, query,combined_qu_ref, n_comb_cells,
                                       clf_cell_pred, ref_celltypes, n_components):
    '''
    Combines the merfish and scRNAseq matrices performs PCA on the combined matrix
    and trains a model to classify cell types by training on the scRNAseq portion of the data. 
    The resulting merfish portion of the data is then classified to cell type probabilities
    ---------------------
    parameters:
    ref: reference single cell Z-scored matrix cells x genes
    query: merfish segmented Z-scored matrix cells x genes
    combined_qu_ref: matrix of zeros the size of the ref and query combined
    n_comb_cells: number of combined cells
    clf_cell_pred: model to classify cell types. Added so we can start from previous parameters
    ref_celltypes: vector of the cell type classification of each scRNAseq cell
    n_components: number of PCs to use
    ---------------------
    return:
        cells_probs: 2d matrix with cells by cell types with a probability of each cell type
        for each cell
    '''
    combined_qu_ref[0:query.shape[0],:] = query
    combined_qu_ref[query.shape[0]:n_comb_cells] = ref
    pca_ref = PCA(n_components=n_components)
    pca_ref.fit(combined_qu_ref)
    embeddings = pca_ref.transform(combined_qu_ref)
    embeddings_qu = embeddings[0:query.shape[0],:]
    embeddings_ref = embeddings[query.shape[0]:,:]
    clf_cell_pred.fit(embeddings_ref, ref_celltypes)
    cells_probs = clf_cell_pred.predict_proba(embeddings_qu)
    return cells_probs

def count_surroundings(cell_assign, num_classes, celltype_pred):
    '''
    counts the cell types of the surrounding pixels for each pixel
    ---------------------
    parameters:
        cell_assign: current assignment of each pixel
        num_classes: the number of cell types
        celltype_pred: predictions of each cells celltype
    returns:
        surr_count: 4d array for each pixel the number of surroundings that are of each cell type
    '''
    classified_assignment = celltype_pred[cell_assign].copy()
    classified_assignment[np.where(cell_assign == -1)] = num_classes-1

    classified_assignment = np.pad(classified_assignment,(1),'constant', constant_values=(-2))
    height = cell_assign.shape[0]; width = cell_assign.shape[1]; depth = cell_assign.shape[2]
    num_squares = height*width*depth
    surroundings = np.array([np.tile(np.arange(0,height),num_squares//height),
                           np.tile(np.arange(0,width),num_squares//width),
                           np.tile(np.arange(0,depth),num_squares//depth)]).T
    surr_count = np.zeros((num_squares,num_classes))
    surroundings = surroundings + 1
    counter = 0
    surr_hei = surroundings[:,0]; surr_wid = surroundings[:,1]; surr_dep = surroundings[:,2];
    ind = np.arange(num_squares)

    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if ((i != 0) | (j != 0) | (k != 0)):
                    counter += 1
                    column_classified = classified_assignment[surr_hei+i,
                                                             surr_wid+j,
                                                            surr_dep+k]

                    surr_count[ind,column_classified] += 1

                    empty_surroundings = np.where(column_classified == -2)[0]
                    surr_count[empty_surroundings, -2] -= 1

    return np.reshape(surr_count, (height, width, depth, num_classes))


def find_empty_pixels(pix):
    '''
    Finds which pixels are empty to be used later to exclude border pix
    ---------------------
    parameters:
        pix: 4d array of pixel identities 
    ---------------------
    return:
        empty_pix: 3d array of pixels 0 if empty 1 if not empty
    '''
    empty_pix = np.sum(pix,axis=3)
    empty_pix[empty_pix > 0] = 1

    return empty_pix


def add_coords_to_cell_matrix(cm, nuc):
    '''
    Adds the cell center coordinates to the cell matrix
    ---------------------
    parameters:
        cm: cell count matrix
        nuc: dataframe of nuclei
    return:
        cm: cell count matrix with coordinates appended to the front
    '''
    cell_centers, nuc_id = get_cell_centers(nuc)
    nuc_xy = pd.DataFrame(np.zeros((cm.shape[0],3)),columns=['x','y','z'])
    nuc_xy.index = cm.index
    nuc_xy.loc[:,:] = cell_centers
    cm = pd.concat([nuc_xy,cm],axis=1)
    
    return cm

def get_cell_centers(nuc):
    '''
    Finds the center of each nucleus
    ---------------------
    parameters:
        nuc: dataframe of nuclei
    ---------------------
    return:
        cell_centers: array of the x,y,z center of each nuclus
        nuc_id: array of the nucleus id in the same order as cell centers
    '''
    cell_centers = []
    nuc_id = []
    for ind in np.unique(nuc.id):
        nuc_id.append(int(ind))
        temp = nuc[nuc.id == ind]
        x_cent, y_cent, z_cent = np.mean(temp.x), np.mean(temp.y), np.mean(temp.z)
        cell_centers.append([x_cent, y_wcent, z_cent])
    cell_centers = np.array(cell_centers)
    nuc_id = np.array(nuc_id)
    return cell_centers, nuc_id
    
def get_matrix_of_cells(pix, cell_assign, nuclei_clusters):
    '''
    Gets the count matrix based on current cell assignment and true input pixels
    ---------------------
    parameters:
        pix: 4d array of pixels to with their gene expression
        cell_assign: current assignments of each pixel to a cell
        nuclei_clusters: dataframe of nuclei points
    ---------------------
    return: cells_mat nxm count matrix where n is number of cells m is number of genes
    '''
    n_gene = pix.shape[3]
    n_cell = len(np.unique(nuclei_clusters.id))
    cells_mat = pd.DataFrame(np.zeros((n_cell,n_gene)),columns = np.arange(n_gene))
    cells_mat.index = np.unique(nuclei_clusters.id).astype(int)
    background = np.zeros(n_gene)
    for i in np.unique(nuclei_clusters.id):
        id_loc = np.where(cell_assign == i)
        cells_mat.loc[i,:] = np.sum(pix[id_loc[0],
                                          id_loc[1],
                                          id_loc[2],:],axis=0)
    
    return cells_mat


def get_number_similar_surroundings(cell_assign):
    '''
    counts the cell types of the surrounding pixels for each pixel
    ---------------------
    parameters:
        cell_assign: current assignment of each pixel
        num_classes: the number of cell types
        celltype_pred: predictions of each cells celltype
    returns:
        surr_count: 4d array for each pixel the number of surroundings that are of each cell type
    '''
    surr_count = np.zeros_like(cell_assign)
    same_count = np.zeros_like(cell_assign)

    surroundings = np.pad(cell_assign,(1),'constant',constant_values=(-2))
    height = cell_assign.shape[0]; width = cell_assign.shape[1]; depth = cell_assign.shape[2]

    surroundings = np.array(surroundings, dtype=int)
    surr_count = np.array(surr_count, dtype=int)
    same_count = np.array(same_count, dtype=int)
    
    c_args = [ndpointer(dtype=ctypes.c_int,flags='C'),
             ndpointer(dtype=ctypes.c_int,flags='C'),
             ndpointer(dtype=ctypes.c_int,flags='C'),
         ctypes.c_int, ctypes.c_int, ctypes.c_int]
    get_num_surr_func_c.get_sur.argtypes = c_args
    get_sur.restype = None

    surroundings = surroundings.ravel().astype(np.int32)
    surr_count = surr_count.ravel().astype(np.int32)
    same_count = same_count.ravel().astype(np.int32)

    
    #c func
    get_sur(surroundings,surr_count,same_count, height, width, depth)
    same_count -= 1

    return surr_count.reshape(cell_assign.shape), same_count.reshape(cell_assign.shape)

def classify_pixels_to_nuclei(locs, nuclei_clust, dist_threshold):
    '''
    Classify each pixel to a nucleus or to nothing (-1)
    ---------------------
    parameters:
        locs: locations of pixels in x, y, z coordinates
        nuclei_clust: dataframe of nuclei spots
        dist_threshold: maximum distance away from nucleus for classification
    '''
    neighbors_classifier = neighbors.NearestNeighbors(1)
    neighbors_classifier.fit( nuclei_clust.loc[:,['x','y','z']].values,nuclei_clust.id)
    l = locs.shape
    new_locs = np.reshape(locs, (l[0]*l[1]*l[2],3))
    cell_assignment = -np.ones(l[0:3])
    dists, predicted = neighbors_classifier.kneighbors(new_locs)
    dists = dists.ravel(); predicted = predicted.ravel()
    predicted = nuclei_clust.id.to_numpy()[predicted]
    predicted[~(dists < dist_threshold)] = -1
    counter = 0
    for i in range(len(cell_assignment)):
        for j in range(len(cell_assignment[i])):
            for k in range(len(cell_assignment[i,j])):
                cell_assignment[i,j,k]=predicted[counter]
                counter += 1
    return cell_assignment.astype(int)

def get_real_pixels(spots, approximate_binsize, genes_mer, pix_shape,dtype=np.float32):
    '''
    Returns the array of pixels using count data instead of smoothed values
    If there is only one spot in a given gene and z plane, it is ignored so
    the number of spots may be slightly more than the sum of the true pixels
    ---------------------
    parameters:
        spots: raw merfish data
        approximate_binsize: the approximated binsize of each histogram cell
        genes_mer: genes that are in the merfish data
        pix_shape: shape of pixels tensor
    ---------------------
    return:
        pix_true: 4d pixel tensor with true count data
    '''
    min_x = np.min(spots.x); max_x = np.max(spots.x);
    min_y = np.min(spots.y); max_y = np.max(spots.y);
    min_z = np.min(spots.z); max_z = np.max(spots.z);

    x_steps = get_real_binsize(spots.x, approximate_binsize)
    y_steps = get_real_binsize(spots.y, approximate_binsize)
    z_steps = get_real_binsize(spots.z, approximate_binsize)
    
    n_x_bins = len(np.arange(min_x,max_x+x_steps,x_steps))
    n_y_bins = len(np.arange(min_y,max_y+y_steps,y_steps))
    
    pix_true = np.zeros(pix_shape,dtype=dtype)
    z_bins = np.arange(min_z, max_z+z_steps+1, z_steps)
    ngene = len(genes_mer)
    
    tic = time()
    for i,gene in enumerate(genes_mer):
        print(gene)
        toc = time()

        spots_temp = spots[spots.gene == gene]
        z_counter = 0
        for z in range(1,len(z_bins),1):
            spots_temp_z = spots_temp[(spots_temp.z >= z_bins[z-1])&
                                     (spots_temp.z < z_bins[z])]
            if spots_temp_z.shape[0] > 1:
                hist = np.histogram2d(spots_temp_z.x,
                                     spots_temp_z.y,
                                     range=[[min_x,max_x],
                                           [min_y,max_y]],
                                     bins = (n_x_bins,
                                            n_y_bins))[0]
                pix_true[:,:,z_counter,i] = hist
            z_counter += 1

    return pix_true
        
    

def get_locations(spots, approximate_binsize):
    '''
    Gets the coordinates for each cell in the pixels
    ---------------------
    parameters:
        spots: merfish raw data
        approximate_binsize: the approximated binsize of each histogram cell
    ---------------------
    return:
        locations: 4d array with x, y, z coordinates for each pixel
    '''
    x_steps = get_real_binsize(spots.x, approximate_binsize)
    y_steps = get_real_binsize(spots.y, approximate_binsize)
    z_steps = get_real_binsize(spots.z, approximate_binsize)
    xs = np.arange(np.min(spots.x),np.max(spots.x)+x_steps,x_steps)
    ys = np.arange(np.min(spots.y),np.max(spots.y)+y_steps,y_steps)
    zs = np.arange(np.min(spots.z),np.max(spots.z)+z_steps,z_steps)
    X, Y, Z = np.mgrid[np.min(spots.x):np.max(spots.x)+x_steps:x_steps,
                      np.min(spots.y):np.max(spots.y)+y_steps:y_steps,
                      np.min(spots.z):np.max(spots.z)+z_steps:z_steps]
    locations = np.zeros((len(xs),len(ys),len(zs),3))
    locations[:,:,:,0] = X
    locations[:,:,:,1] = Y
    locations[:,:,:,2] = Z
    return locations


def fast_de_all_spots(spots, approximate_binsize,
                       bandwidth):
    '''
    Runs psuedo-kde for all genes
    ---------------------
    parameters:
        spots: merfish raw data
        approximate_binsize: the approximated binsize of each histogram cell
        bandwidth: how far away to get information from for kde
    ---------------------
    return:
        kde_data: 4d array of all kde data for every gene (4th dimension)
    '''
    positions, x_shape = get_positions_for_kde(spots, approximate_binsize)
    kde_data = np.zeros((x_shape[0],x_shape[1],x_shape[2],len(np.unique(spots.gene))))
    for i,gene in enumerate(np.unique(spots.gene)):
        print(gene, i)
        temp = spots[spots.gene == gene]
        kde_data[:,:,:,i] = fast_kde_spot(temp, positions,
                                          approximate_binsize, bandwidth,
                                          x_shape)
    return kde_data

def fast_kde_spot(spots, positions, approximate_binsize,
                  bandwidth,x_shape):
    '''
    Wrapper for running fast_kde_with_knn for the spots
    ---------------------
    parameters:
        spots: merfish raw data
        positions: center points for kde smoothing
        approximate_binsize: the approximated binsize of each histogram cell
        x_shape: final shape of the 3d array for the smoothed kde
    return:
        spot_dense: 3d array with smoothed kde vlaues
    '''
    coords = spots.loc[:,['x','y','z']].to_numpy()
    spot_dense = np.reshape(fast_kde_with_knn(positions, coords,
                                              bandwidth),x_shape)
    return spot_dense

def kde_nuclei(spots, nuclei,
               approximate_binsize, bandwidth):
    '''
    Get the smoothed density of the nuclei
    ---------------------
    parameters:
        spots: dataframe of the merfish spots
        nuclei: dataframe of the nuclei points
        approximate_binsize: approximate binsize in microns
        bandwidth: number of neighbors to look for
    ---------------------
    return:
        nuc_dense: 3d array with the smoothed nuclei density
    '''
    
    positions, x_shape = get_positions_for_kde(spots, approximate_binsize)
    coords = nuclei.loc[:,['x','y','z']].to_numpy()

    print('getting density for nuclei')
    nuc_dense = np.reshape(fast_kde_with_knn(positions, coords,
                                             bandwidth,1),x_shape)
    nuc_dense *= nuclei.shape[0]
    nuc_dense /= np.mean(nuc_dense)
    return nuc_dense

def get_positions_for_kde(spots, approximate_binsize):
    '''
    Creates the grid to get the positions where to find kde 
    ---------------------
    parameters:
        spots: raw merfish data
        approximate_binsize: approximated binsize of the histogram cell
    ---------------------
    return:
        positions: positions with coordinates of where to find kde
        x_shape: the shape of the final 3d array 
    '''
    
    x_steps = get_real_binsize(spots.x, approximate_binsize)
    y_steps = get_real_binsize(spots.y, approximate_binsize)
    z_steps = get_real_binsize(spots.z, approximate_binsize)
    xs = np.arange(np.min(spots.x),np.max(spots.x)+x_steps,x_steps)
    ys = np.arange(np.min(spots.y),np.max(spots.y)+y_steps,y_steps)
    zs = np.arange(np.min(spots.z),np.max(spots.z)+z_steps,z_steps)

    X, Y, Z = np.mgrid[np.min(spots.x):np.max(spots.x)+x_steps:x_steps,
                      np.min(spots.y):np.max(spots.y)+y_steps:y_steps,
                      np.min(spots.z):np.max(spots.z)+z_steps:z_steps]
    
    positions = np.vstack([X.ravel(),Y.ravel(),Z.ravel()]).T
    return positions, X.shape

def fast_kde_with_knn(positions, coords, nneigh):
    '''
    Pseudo-KDE by dividing the number of points near the point of interest
    by the volumne take to get to that number
    ---------------------
    parameters:
        positions: vector of coordinates in x,y,z to find kde on
        coords: coordinates of the points to be smoothed by kde
        nneigh: number of neighbors to use, it is a proxy for bandwidth, or 
                distance to pull information from
    return:
        kde vec: vector of the smoothed kde values for each location in positions
    '''
    nneigh = min(nneigh, coords.shape[0])    
    nbrs = neighbors.NearestNeighbors(n_neighbors=nneigh, algorithm='kd_tree',
                                     n_jobs=24).fit(coords)
    distances, indices = nbrs.kneighbors(positions)
    denom = ((4/3*3.14))*distances[:,nneigh-1]**3
    denom = np.maximum(denom,1e-1)
    return nneigh/denom


def get_real_binsize(one_direction, approx_binsize):
    '''
    We approximate the binsize in microns, but because the actual range may not be divisible by the 
    approximated binsize we need to change the true binsize
    ---------------------
    parameters:
        one_direction: the vector of coordinates for x, y, or z
        approx_binsize: approximate binsize in microns
    ---------------------
    return:
        actual binsize: the range/number of bins
    
    '''
    one_range = get_range(one_direction)
    nbins = np.ceil(one_range/approx_binsize) #so we can have equal sized bins
    return one_range/nbins

def get_range(array):
    '''
    Gets the range of the coordinates for either x, y, or z
    ---------------------
    parameters:
        array: numpy array of x, y, or z values
    ---------------------
    return:
        float: the range of the coordinates max-min
    '''
    return(np.max(array)-np.min(array))

def shrink_window(df, x_min, x_max, y_min, y_max):
    '''
    Shrinks the window down to a smaller size
    ---------------------
    parameters:
        df: dataframe to shrink
        x_min: lower bound x
        x_max: upper bound x
        y_min: lower_bound y
        y_max: upper_bound y
    ---------------------
    return:
        new_df: shrunk dataframe
    '''
    new_df = df[(df.x > x_min) &
               (df.x < x_max) &
               (df.y > y_min) &
               (df.y < y_max)]
    return new_df

def plot_segmentation(assignment, cmap='nipy_spectral'):
    '''
        Plots segmentation map as the max value through the z-stack
        ---------------------
        parameters:
            assignment: 3d array of the cell segmentation map
            cmap: matplotlib color map to use
    '''

    #rearange the cell ids so the coloring is more spread out
    colors = np.unique(assignment)
    copy_assign = assignment.copy()
    np.random.shuffle(colors)
    colors[colors == -1] = colors[0]
    colors[0] == -1
    for i,c in enumerate(np.unique(copy_assign)):
        #skip -1 in the segmentation map
        if c != -1:
            copy_assign[copy_assign == c] = colors[i]

    #plot the max value through the z-stack
    plt.imshow(np.max(copy_assign, axis=2),
               cmap=cmap)

def reclassify_squares(pix, pixl_true,
                       cell_matrix, nuc,
                       cell_assign,  sc_ref,
                       sc_ref_celltypes, all_genes,
                       locs, clf_cell,
                       pct_train=0.1, border_other_threshold=5,
                       border_same_threshold=2,
                       outer_max=1, inner_max=5,
                       most_inner_max=5, dist_threshold=2, dist_scaling=5,
                       anneal_param=0.05, flip_thresh=0.1,
                       nlayer=3, first_epochs=25, second_epochs=15,
                       lrs=[1e-3, 1e-4], l1_reg=1e-3):
    '''
    Method to flip pixels from one cell to another or to no cell assignment to improve cell segmentation
    High level description: 
        a) Classify cells to a cell type.
        b)Train on a subset of the pixels to build a pixel level
            classifier for determining the celltype identity  
        c) Flip border pixels acording to their predictions using the model in b.
        Keep switching between a, b, and c flipping pixels and retraining the models
    ---------------------
    parameters:
        pix: 4d tensor. 4th dimension is gene expression (kde) vector of each pixel
        pixl_true: 4d tensor. 4th dimension is gene expression (counts) vector of each pixel
        cell_matrix: voronoi segmented digital gene expression matrix cells x genes
        nuc: data frame with x,y,z coordinates of nuclei pixels with nuclei IDs
        cell_assign: 3d tensor. each pixel has it's current nuclei classification or none
        sc_ref: scRNAseq reference matrix cells x genes
        sc_ref_celltypes: vector of cell types for each scRNAseq cell
        all_genes: genes in merfish data
        locs: 4d tensor. 4th dimension is an x,y,z vector of the coordinates of the center of that pixel
        clf_cell:  neural network based cell type predictor
        pct_train: percentage of pixels to train on for pixel classifier (default: 0.1)
        border_other_threshold: how many pixels need to belong to the other cell to flip to that cell (default: 5)
        border_same_threshold: border_other_threshold: how many pixels need to belong to the same cell to flip to that cell (default: 2)
        outer_max: numbegr of iterations of the outer loop (cell classification (a)) (default: 1)
        inner_max: number of iterations of the inner loop (pixel training (b)) (default: 5)
        most_inner_max: number of iterations of the flipping pixels loop (pixel flip (c)) (default: 5)
        dist_threshold: distance from edge of nucleus to ensure a cell belongs to that nucleus. (default: 2)
        dist_scaling: amount of decay of probabilities for flipping as you move away from a nucleus of
            interest. The probability decreases by half every dist_threshold*dist_scaling (default: 5)
        annealing_param: Parameter to decrease the probabalistic component of pixel and cell classification.
            Every iteration the highest probability is multiplied by 1+annealing_param*n_iteration (default: 0.5)
        flip_thresh: Pixel probabilities before this value get set to 0. (default: 0.1)
        nlayer: number of intermediate layers in the pixel classifier (default: 3)
        first_epochs: number of epochs for training after the pixel classifier was initialized (default: 25)
        second_epochs: number of epochs for training on subsequent rounds of training (default: 15)
        lrs: list of learning rates for training pixel classifier (default: [1e-3, 1e-4])
        l1_reg: l1 regularization parameter for neural network 
    return:
        cell_assign: 3d tensor giving the nuclei assignment of each pixel 
    '''
    # copy of original assignment for later use
    copy_cell_assign = cell_assign.copy()

    # name change. Get rid of later
    genes_to_use_prediction = all_genes

    # gets the genes we need for cell type prediction
    gene_subset_indices = []
    for i in all_genes:
        if i in genes_to_use_prediction:
            gene_subset_indices.append(True)
        else:
            gene_subset_indices.append(False)

    # train nuclei knn classifier for later use
    nuc_clf = neighbors.NearestNeighbors(n_neighbors=10).fit(nuc.loc[:, ['x', 'y', 'z']],
                                                             nuc.id)

    # get count of surroundings that are the same and different
    surround_count, same_cts = get_number_similar_surroundings(cell_assign)

    # get number of cell types
    n_celltype = len(np.unique(sc_ref_celltypes))

    pix_shape = pix.shape
    x_max, y_max, z_max = pix_shape[0]-1, pix_shape[1]-1, pix_shape[2]-1

    # name change from before need to clean up
    cp_grid = pix

    num_iterations_outer = 0
    np.seterr(invalid='raise')
    square_param_diff_vec = []

    prediction_mean = []
    p_weight = None
    p_mean = None
    percent_flipped = []
    logi_param_diff = []

    # center and scale sc ref
    #sc_ref.loc[:,:] = scale(sc_ref, axis=0)

    overlapping_genes_for_merfish_map = np.isin(all_genes,
                                                sc_ref.columns)

    n_combined_cells = cell_matrix.shape[0]+sc_ref.shape[0]
    combined_cells = np.zeros((n_combined_cells,
                               np.sum(overlapping_genes_for_merfish_map)))

    clf_log = pixel_nn_classifier(sc_ref,
                                  sc_ref_celltypes,
                                  nlayer,
                                  l1_reg)

    n_iterations = 0
    n_changed = []
    n_changed_overall = []

    while(num_iterations_outer < outer_max):
        cells_matrix = get_matrix_of_cells(
            pixl_true, cell_assign, nuc).to_numpy()

        non_empty_cell_locs = np.where(np.sum(cells_matrix, axis=1) > 100)[0]
        cells_matrix = scale(cells_matrix, axis=1)
        cells_matrix = scale(cells_matrix, axis=0)
        tic = time()

        # combines the mer and sc_ref matrices and builds a classifier for celltype
        # based on sc_ref
        print('finding celltypes')
        cells_probs = clf_cell.predict(cells_matrix)

        toc = time()
        print('time to get celltypes', toc-tic)

        prediction_mean.append(np.mean(np.max(cells_probs, axis=1)))

        # adding multiply the max prob by 1+n_iteration*annealing_param
        max_pred = np.argmax(cells_probs, axis=1)

        cells_probs[np.arange(len(max_pred)),
                    max_pred] *= 1+n_iterations*anneal_param
        cells_probs /= np.sum(cells_probs, axis=1, keepdims=True)

        # get the identity of the predicted cell types
        groupings = np.argmax(cells_probs, axis=1)

        #groupings = t_cell

        last_param = None

        toc = time()
        #print('time to find cell types ',toc-tic)

        num_iterations_inner = 0
        past_square_param = None
        while(num_iterations_inner < inner_max):
            #print('inner iterations:',num_iterations_inner)
            flat_assign = np.ravel(cell_assign)
            group_labels = groupings[flat_assign]

            # the empty cell type will be index of number of celltypes
            group_labels[flat_assign == -1] = n_celltype

            # random selection of pixels to use for training the model
            subset_indices = np.random.choice(np.arange(0,
                                                        flat_assign.shape[0], dtype=int),
                                              size=int(pct_train*len(group_labels)))
            tic = time()
            merged_pix_info = cp_grid

            merged_pix_shape = merged_pix_info.shape
            merged_pix_reshaped = np.reshape(
                merged_pix_info, (merged_pix_shape[0]*merged_pix_shape[1]*merged_pix_shape[2], merged_pix_shape[3]))
            group_labels_mat = np.reshape(
                group_labels, (merged_pix_shape[0], merged_pix_shape[1], merged_pix_shape[2]))

            sub_merged_pix = merged_pix_reshaped[subset_indices, :]
            sub_group_labels = group_labels[subset_indices]

            # remove non cell pixels from training
            sub_merged_pix = sub_merged_pix[sub_group_labels != n_celltype, :]
            sub_group_labels = sub_group_labels[sub_group_labels != n_celltype]

            # train a model to see what a pixel of a certain kind looks like
            if ((num_iterations_inner == 0)):
                clf_log = train_nn_classifier(sub_merged_pix, sub_group_labels, clf_log,
                                              first_epochs, lrs)
            else:
                clf_log = train_nn_classifier(sub_merged_pix, sub_group_labels, clf_log,
                                              second_epochs, lrs[-1])

            toc = time()
            print('time to train ', len(subset_indices), 'samples ', toc-tic)
            num_iterations_most_inner = 0
            while(num_iterations_most_inner < most_inner_max):
                # find border indices where they are not empty, and have more neighbors than the border
                # threshold
                border_indices = np.where((surround_count >= border_other_threshold) &
                                          (same_cts >= border_same_threshold))

                border_indices_mat = np.stack(
                    [border_indices[0], border_indices[1], border_indices[2]], axis=1)

                tic = time()
                # predict the probability a border pixel is from each class
                predictions = clf_log.predict(
                    merged_pix_info[border_indices[0], border_indices[1], border_indices[2], :])

                predictions[predictions <= flip_thresh] = 0

                ngene = merged_pix_info.shape[3]
                ms = merged_pix_info.shape
                npix = ms[0]*ms[1]*ms[2]

                # adding multiply the max prob by 1+n_iteration*annealing_param
                max_pred = np.argmax(predictions, axis=1)

                predictions[np.arange(len(max_pred)),
                            max_pred] *= 1+n_iterations*anneal_param

                if predictions.shape[1] < (n_celltype):
                    diff = np.setdiff1d(
                        np.arange(n_celltype+1), np.unique(group_labels[subset_indices]))
                    diff = np.sort(diff)
                    for missing in diff:
                        predictions = np.insert(predictions, missing, np.repeat(
                            0, predictions.shape[0]), axis=1)
                    print('MISSSING ROW INSERTED!')

                toc = time()
                print('time to predict ', len(
                    border_indices[0]), 'samples', toc-tic)
                # get the number of each cell type pixel surrounding each border pixel

                # some house keeping for the next step by padding arrays
                border_indices_mat += 1
                cell_assign = np.pad(
                    cell_assign, (1), 'constant', constant_values=(-2))

                locs = np.pad(locs, ((1, 1), (1, 1), (1, 1), (0, 0)),
                              'constant', constant_values=(-2))

                bord_x, bord_y, bord_z = border_indices_mat[:,
                                                            0], border_indices_mat[:, 1], border_indices_mat[:, 2]
                surroundings = np.zeros(
                    (border_indices_mat.shape[0]), dtype=int)
                group_key = np.zeros((border_indices_mat.shape[0]), dtype=int)
                predic_num = np.zeros_like(predictions)
                predic_probs = np.zeros(
                    (predictions.shape[0], predictions.shape[1]))
                tic = time()

                pixels_to_nuc_dist_vec, pixels_to_nuclei_vec = nuc_clf.kneighbors(locs[bord_x,
                                                                                       bord_y,
                                                                                       bord_z, :])

                pixels_to_nuclei_vec = nuc.id.to_numpy()[pixels_to_nuclei_vec]
                toc = time()
                print('time to predict nuclei distance:', toc-tic)

                tic = time()

                # this creates a matrix where each row is the surrounding cell ids for that border pixel
                surroundings = np.zeros((len(bord_x), 27))
                sub_counter = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            surroundings[:, sub_counter] = cell_assign[bord_x+i,
                                                                       bord_y+j,
                                                                       bord_z+k].copy()
                            sub_counter += 1

                tic = time()

                input_surr = (surroundings.__array_interface__[
                              'data'][0]+np.arange(surroundings.shape[0])*surroundings.strides[0]).astype(np.uintp)
                input_nuc = (pixels_to_nuclei_vec.__array_interface__['data'][0]+np.arange(
                    pixels_to_nuclei_vec.shape[0])*pixels_to_nuclei_vec.strides[0]).astype(np.uintp)
                input_dist = (pixels_to_nuc_dist_vec.__array_interface__['data'][0]+np.arange(
                    pixels_to_nuc_dist_vec.shape[0])*pixels_to_nuc_dist_vec.strides[0]).astype(np.uintp)
                dist_mat = np.zeros(
                    (surroundings.shape[0], 27), dtype=np.float64)
                input_dist_mat = (dist_mat.__array_interface__[
                                  'data'][0]+np.arange(dist_mat.shape[0])*dist_mat.strides[0]).astype(np.uintp)

                # C function
                get_d(input_surr,
                      input_nuc,
                      input_dist,
                      ctypes.c_int(surroundings.shape[0]),
                      ctypes.c_int(pixels_to_nuclei_vec.shape[1]),
                      input_dist_mat)

                toc = time()

                tic = time()
                for sub_counter in range(27):
                    surro = surroundings[:, sub_counter].copy().astype(int)

                    num_surroundings = len(surro)

                    # get dist vec based on the first occurence of the surroundings
                    # in the nearest neighbors vec from pix to nuclei
                    dist_vec = dist_mat[:, sub_counter]

                    reduced_dist = dist_vec - dist_threshold
                    reduced_dist[reduced_dist <= 0] = 1e-3

                    s = (dist_scaling*dist_threshold)/2
                    scale_vec = np.divide(s, reduced_dist)
                    scale_vec[dist_vec < dist_threshold] = 10
                    scaled_final = np.minimum(10, scale_vec)

                    scaled_final[np.where(surro == -1)] = 1
                    scaled_final[np.where(surro == -2)] = 1

                    group_key = groupings[surro].copy()
                    group_key[np.where(surro == -1)] = -1
                    group_key[np.where(surro == -2)] = -2
                    non_neg = np.where(group_key >= 0)
                    predic_num[non_neg, group_key[non_neg]] += 1
                    predic_probs[non_neg,
                                 group_key[non_neg]] += scaled_final[non_neg]

                    predic_num[np.where(predic_num == 0)] = 1

                predic_probs = np.divide(predic_probs, predic_num)

                predictions = np.multiply(predictions, predic_probs)

                pred_sum = np.sum(predictions, axis=1, keepdims=1)
                pred_sum[pred_sum == 0] = 1
                predictions /= pred_sum

                toc = time()

                number_of_border_pixels = border_indices_mat.shape[0]

                locs_to_flip = locs[border_indices_mat[:, 0],
                                    border_indices_mat[:, 1], border_indices_mat[:, 2], :]

                # remove 0 pad from locs
                ls = locs.shape
                locs = locs[1:ls[0]-1, 1:ls[1]-1, 1:ls[2]-1, :]

                flips_arg_bool = (predictions.cumsum(
                    1) > np.random.rand(predictions.shape[0])[:, None])

                flips_arg = flips_arg_bool.argmax(1)

                # set flip args that have prediction prob 0 to n_celltype
                flips_arg[np.sum(flips_arg_bool, axis=1) == 0] = n_celltype
                flips_arg[np.sum(predictions, axis=1) == 0] = n_celltype

                cells_to_flip = flips_arg

                borders_to_flip = border_indices_mat

                tic = time()

                bord_x, bord_y, bord_z = borders_to_flip[:,
                                                         0], borders_to_flip[:, 1], borders_to_flip[:, 2]
                old_id = cell_assign[bord_x,
                                     bord_y,
                                     bord_z].copy()

                surroundings = np.zeros(
                    (border_indices_mat.shape[0], 3**border_indices_mat.shape[1]), dtype=int)
                group_key = np.zeros(
                    (border_indices_mat.shape[0], 3**border_indices_mat.shape[1]), dtype=int)
                matching_regions = np.zeros(
                    (border_indices_mat.shape[0], 3**border_indices_mat.shape[1]), dtype=np.float32)
                changed_surround_count = np.zeros(
                    (border_indices_mat.shape[0], 3**border_indices_mat.shape[1]), dtype=int)
                loc_counter = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            surroundings[:, loc_counter] = cell_assign[bord_x+i,
                                                                       bord_y+j,
                                                                       bord_z+k].copy()

                            group_key[:, loc_counter] = groupings[surroundings[:, loc_counter]].copy(
                            )
                            group_key[:, loc_counter][np.where(
                                surroundings[:, loc_counter] == -1)] = n_celltype
                            matching_regions[:, loc_counter] = group_key[:,
                                                                         loc_counter] == cells_to_flip
                            loc_counter += 1

                match_dist = matching_regions * dist_mat
                match_dist[match_dist == 0] = 1e3
                new_cell_loc = np.argmin(match_dist, axis=1)

                new_id = surroundings[np.arange(
                    0, surroundings.shape[0]), new_cell_loc].copy()
                new_id[np.sum(matching_regions, axis=1) == 0] = -1
                new_id[np.sum(flips_arg_bool, axis=1) == 0] = -1
                new_id[new_id == -2] = -1

                n_changed_this_round = len(
                    new_id) - np.sum(np.equal(old_id, new_id))
                n_changed.append(n_changed_this_round)
                n_changed_overall.append(n_changed_this_round)
                if len(n_changed) > 10:
                    n_changed = n_changed[1:]

                cell_assign[bord_x, bord_y, bord_z] = new_id

                same_and_different = surroundings == new_id.reshape(
                    surroundings.shape[0], 1)
                different_count = np.sum(same_and_different == 0, axis=1)
                borders_to_flip -= 1

                # remove 0 pad from cell assignment
                cs = cell_assign.shape
                cell_assign = cell_assign[1:cs[0]-1, 1:cs[1]-1, 1:cs[2]-1]
                cell_assign[copy_cell_assign != -
                            1] = copy_cell_assign[copy_cell_assign != -1].copy()

                bord_x, bord_y, bord_z = borders_to_flip[:,
                                                         0], borders_to_flip[:, 1], borders_to_flip[:, 2]

                surround_count, same_cts = get_number_similar_surroundings(
                    cell_assign)

                old_cells = groupings[old_id].copy()
                old_cells[np.where(old_id == -1)] = n_celltype

                new_cells = groupings[new_id].copy()
                new_cells[np.where(new_id == -1)] = n_celltype

                percent_flipped.append(
                    np.sum(np.equal(new_id, old_id))/number_of_border_pixels)

                toc = time()
                print('time to flip pixels ', toc-tic)
                n_iterations += 1
                num_iterations_most_inner += 1
            num_iterations_inner += 1
        num_iterations_outer += 1

    # compute counts matrix
    cells_matrix = get_matrix_of_cells(pixl_true, cell_assign, nuc)

    return cell_assign, cells_matrix, groupings

def watershed_nuclei(pix, nuclei, locations):
    ls = locations.shape
    raveled_locs = locations.reshape((ls[0]*ls[1]*ls[2],3))
    square_ids = np.arange(0,raveled_locs.shape[0])
    locs_clf = neighbors.KNeighborsClassifier(1,
                                             n_jobs=12).fit(raveled_locs, square_ids)
    
    predicted_locs = locs_clf.predict(nuclei.loc[:,['x','y','z']])
    nuclei_ids = np.zeros(len(square_ids))
    nuclei_ids[predicted_locs] = nuclei.id+1
    nuclei_ids = nuclei_ids.reshape((ls[0],ls[1],ls[2]))
    
    pix_dens = np.log2(np.sum(pix,axis=3)+1)
    image = np.zeros_like(pix_dens)
    image[pix_dens > pix_thresh] = 1
    labels = watershed(pix_dens, nuclei_ids,
                      watershed_line = True,
                      mask = image, compactness=10) - 1
    return labels

def create_celltype_classifier(sf, sc, nlayers=2, l1_reg=1e-3,
                               epochs=20, lrs=[5e-3, 5e-4],
                               test_size=0.25):
    '''
        Creates a cell type classifier and trains the model based on a neural network
        ---------------------
        parameters:
            sf: single cell reference dataset
            sc: single cell reference cell types. (must be int)
            nlayers: number of intermediate layers in network (default: 2)
            l1_reg: l1 regularization parameter (default: 1e-3)
            epochs: number of epochs to train for, per learning rate cycle (default: 20)
            lrs: learning rates. Must be a list of learning rates (defaults: [5e-3,5e-4])
            test_size: percentage of dataset to use for validation (default: 0.25)
        returns:
            clf_cell: Trained cell type classifier

    '''

    ncelltype = len(np.unique(sc))

    input_dim = sf.shape[1]
    input_vec = Input(shape=(input_dim,))
    x = Dense(input_dim*3, activation='tanh',
              activity_regularizer=l1(l1_reg))(input_vec)
    x = BatchNormalization()(x)
    for i in range(nlayers-1):
        x = Dense(input_dim*3, activation='tanh',
                  activity_regularizer=l1(l1_reg))(x)
        x = BatchNormalization()(x)
    out = Dense(ncelltype, activation='softmax',
                activity_regularizer=l1(l1_reg))(x)

    clf_cell = Model(input_vec, out)
    clf_cell.summary()

    scaled_ref = scale(sf, axis=1)
    scaled_ref = scale(scaled_ref, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(scaled_ref, sc,
                                                        test_size=test_size,
                                                        random_state=0)

    for lr in lrs:
        adam = Adam(learning_rate=lr)
        clf_cell.compile(optimizer='Adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        clf_cell.fit(X_train, y_train,
                     validation_data=(np.array(X_test), np.array(y_test)),
                     epochs=20,
                     batch_size=64,
                     use_multiprocessing=True)

    return clf_cell
    
def pixel_nn_classifier(mp,sc,nlayer, l2_reg):
    ncelltype = len(np.unique(sc))
    input_dim = mp.shape[1]
    input_vec = Input(shape=(input_dim,))
    x = Dense(input_dim*2,activation='tanh',
             activity_regularizer=l1(l2_reg))(input_vec)
    x = BatchNormalization()(x)
    for i in range(1,nlayer):
        x = Dense(input_dim*((2)**(i)),activation='tanh',
             activity_regularizer=l1(l2_reg))(x)
        x = BatchNormalization()(x)
        
    out = Dense(ncelltype, activation='softmax',
               activity_regularizer=l1(l2_reg))(x)

    clf= Model(input_vec, out)
    clf.summary()
    return clf
    
def train_nn_classifier(mp,sc,clf,epo,lrs):
    X_train, X_test, y_train, y_test = train_test_split(mp, sc,
                                                   test_size = 0.2,random_state = 0)
    for lr in lrs:
        adam = Adam(learning_rate = lr)
        clf.compile(optimizer = 'Adam',
                         loss = 'sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        clf.fit(X_train,y_train,
               epochs = epo,
               batch_size = 64,
               validation_data=(X_test,y_test),
               use_multiprocessing=True)
    return clf

def get_centers(vec):
    '''
    # gets the center of each pixel
    # input:
    #    vec: the aranged vector of the edge of each pixel
    # output:
    #    ctr: vector of len(vec)-1 that has the center of each pixel's coordinates
    '''
    ctr = []
    for i in range(len(vec)-1):
        ctr.append(np.mean([vec[i],vec[i+1]]))
    return np.array(ctr)

def transform_to_stage_coordinates(spt, metadata):
    '''
    # transforms the nuclei to stage coordinates and creates a dataframe
    # that shows the location of each nuclei assignment
    # input:
    #    nuc_assign: 3d tensor of the nuclei assignments
    #    XY: a vector of the center of each pos [x,y]
    #    zstack: vector of the z indices 
    #    pix_size: size of each pixel from metadata 
    #    img_shape: 3d shape of the image with nuclei assignments
    # output:
    #    df_nuc: dataframe with the nuclei locations
    '''
    new_spots = np.zeros((spt.shape[0],4))
    spt_counter = 0
    genes = []
    for pos in np.unique(spt.posname):
        print(pos)
        temp_spt = spt[spt.posname == pos]
        genes += list(temp_spt.gene.to_numpy())
        subset_metadata = metadata[metadata.Position == pos]
        XY = subset_metadata.XYbeforeTransform.to_numpy()[0]
        pix_size = subset_metadata.PixelSize.to_numpy()[0]

        x = XY[0]
        y = XY[1]
        
        for s in zip(temp_spt.centroid,temp_spt.z):
            ctr_x = (s[0][1]-1024) * pix_size
            ctr_y = (s[0][0]-1024) * pix_size
            ctr_z = s[1] * 0.4
            new_spots[spt_counter,1:] = [ctr_x+x,ctr_y+y,ctr_z]
            spt_counter += 1
    
    df_spt = pd.DataFrame(new_spots)
    df_spt.columns = ['gene','x','y','z']
    df_spt.gene = np.array(genes)
    
    return df_spt

c_args = [ndpointer(dtype=np.uintp,ndim=1,flags='C'),
         ndpointer(dtype=np.uintp,ndim=1,flags='C'),
         ndpointer(dtype=np.uintp,ndim=1,flags='C'),
         ctypes.c_int, ctypes.c_int,
         ndpointer(dtype=np.uintp,ndim=1,flags='C')]

dist_func_c = ctypes.CDLL(path_to_file+"/get_distances.so")
get_d = dist_func_c.get_distances
dist_func_c.get_distances.argtypes = c_args
get_d.restype = None
get_num_surr_func_c = ctypes.CDLL(path_to_file+"/get_number_similar_surroundings.so")
get_sur = get_num_surr_func_c.get_sur

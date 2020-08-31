import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from random import randint
from collections import Counter
import matplotlib.patches as mpatches
import pickle as pkl
import json
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from umap import UMAP
from sklearn.model_selection import train_test_split

class merfish_data_generator:
    '''
        Class to generate merfish data, all values are in microns
        --------------------
        parameters:
            celltypes: list of the unique cell types (default: None)
            genes: list of the genes used for merfish data (default: None)
            celltype_props: list of the cell type proportions should be in the same order as
                celltypes (default: None)
            density_range: range of areas, and how dense to make them (default: [1])
            dist_between_cell_centers: range of values of how far apart
                to make them (default: [30])
            cell_shape_regularity: how round the cell should be (default: 1)
            dge: digital gene expression matrix celltypes x genes (default: None)
            noise: overlap between cells count data (default: 0)
            heterogeneity: how heterogeneous to make the cell type distribution (default: 1)
            grid_shape: x, y, z sizze of the grid
            nuclei_size_range: variability in the size of the nuclei (default: [0.2])
            distance_between_cells: adds to the overall distance between cells
                negative values squish cells closer together (default: 0)
            subtype: Boolean, indicating whether or not the current cells are 
                                subtypes
        --------------------
        functions:
            generate_grid: Generates the xyz grid for where the cell centers are
            generate_cell_centers: Generates a dataframe with the cell centers in xyz coords
            assign_pixels_to_cells: Assigns the pixels to the cells based on
                modified voronoi from the center of the cell w/ max distance
            plot_true: Plots the cell ID grid in a single z_stack value
            plot_celltypes: Plots the celltype grid in a single z_stack value
            generate_nuclei: Generates the nuclei randomly within a cell
            add_dge: Adds the DGE matrices to the object both should be celltypes x genes
                Can be used to add a self curated dge instead of computing one
            compute_dge: Computes and adds the DGE  matrices to the object. It should be in celltypes x genes
                Also computes cell type proportions
            classify_celltypes: Adds celltypes to the cells based on the cell type proportions
            generate_merfish_dge: Generates the merfish dge for the cells, based on their cell type and the dge from
                single cell data
            place_transcripts: Places the spots within the cell
    '''
    def __init__(self, celltypes = None,
                genes = None,
                 celltype_props = None,
                density_range = [1],
                dist_between_cell_centers = [30],
                cell_shape_regularity = 1,
                dge = None,
                noise = 0,
                heterogeneity = 1,
                grid_shape = (200,200,50),
                nuclei_size_range = [0.2],
                distance_between_cells = 0,
                subtype = False):
        self.celltypes = celltypes
        self.genes = genes
        self.density_range = density_range
        self.dist_between_cell_centers = dist_between_cell_centers
        self.cell_shape_regularity = cell_shape_regularity
        self.celltype_props = celltype_props
        self.dge = dge
        self.noise = noise
        self.heterogeneity = 1
        self.grid_shape = grid_shape
        #percentage of cell size
        self.nuclei_size_range = nuclei_size_range
        self.x_max = grid_shape[0]; self.x_min = 0
        self.y_max = grid_shape[1]; self.y_min = 0
        self.z_max = grid_shape[2]; self.z_min = 0
        self.distance_between_cells = distance_between_cells
        
        self.cell_centers = None
        self.true_map = None
        
        self.subtype = subtype
        
        
    def generate_grid(self, space_between):
        '''
            Generates the xyz grid for where the cell centers are
            --------------------
            parameters:
                space_between: space in microns between nuclei
            --------------------
            returns:
                x_coord: x coordinate vector of nuclei
                y_coord: y coordinate vector of nuclei
                z_coord: z coordinate vector of nuclei
        '''
        xs = np.arange(0,self.x_max, space_between)
        ys = np.arange(0,self.y_max, space_between)
        zs = np.arange(0,self.z_max, space_between)        
        x_coord, y_coord, z_coord = np.meshgrid(xs,ys,zs)

        return x_coord, y_coord, z_coord
        
        
    def generate_cell_centers(self):
        '''
            Generates a dataframe with the cell centers in xyz coords
            --------------------
            adds:
                self.cell_centers: dataframe with xyz coordinates
                self.cell_ids: the identification number of each cell
        '''
        distance_between_nuclei = np.mean(self.dist_between_cell_centers)+self.distance_between_cells
        std_distance_between_nuclei = np.std(self.dist_between_cell_centers)

        x_coord, y_coord, z_coord = self.generate_grid(distance_between_nuclei)
        
        gs = x_coord.shape
        
        x_rand = np.random.randn(gs[0],gs[1],gs[2])*std_distance_between_nuclei
        y_rand = np.random.randn(gs[0],gs[1],gs[2])*std_distance_between_nuclei
        z_rand = np.random.randn(gs[0],gs[1],gs[2])*std_distance_between_nuclei
        
        x_coord += x_rand; y_coord += y_rand; z_coord += z_rand
        
        cell_centers = pd.DataFrame({'x':x_coord.ravel(),
                                     'y':y_coord.ravel(),
                                     'z':z_coord.ravel()})

        self.cell_centers = cell_centers
        cell_ids = np.arange(cell_centers.shape[0])
        self.cell_ids = cell_ids
        
    def assign_pixels_to_cells(self,
                               pixels_per_micron = .5,
                               max_dist = 20,
                              noise_in_dist = 0,
                              min_pix_count = 30):
        '''
            Assigns the pixels to the cells based on modified voronoi from the
            center of the cell w/ max distance
            --------------------
            parameters:
                pixels_per_micron: measure for the resolution of the grid. Higher values increase
                    resolution, but increase run time and memory requirements (default: .5)
                max_dist: maximum distance away from the center for voronoi segmentation (default: 20 microns)
                noise_in_dist: adds uneven edge effects the the nuclei (default: 0)
                min_pix_count: minimum pixels assigned to a cell for a cell to be considered a cell
            --------------------
            adds:
                self.cell_ids: removes the cell ids with < min_pix_count pixels assigned to it
                self.cell_centers: removes the centers of cells with < min_pix_count
                self.true_map: the 3d map of pixels -1 is no id, otherwise  the number is the cell id
        '''
        if self.cell_centers is None:
            self.generate_cell_centers()

        x_coord, y_coord, z_coord = self.generate_grid(1/pixels_per_micron)
        self.pix_per_micron = pixels_per_micron
        
        #change this later
        max_dist = np.mean(self.dist_between_cell_centers)/2
        
        clf_cell_center = KNeighborsClassifier(n_neighbors=1,
                                               algorithm='kd_tree').fit(self.cell_centers.to_numpy(),
                                                                         self.cell_ids)

        dist, cell_id = clf_cell_center.kneighbors(np.array([x_coord.ravel(),
                                                            y_coord.ravel(),
                                                            z_coord.ravel()]).T)
        dist = dist.ravel(); cell_id = cell_id.ravel()
        dist += np.random.randn(len(dist)) * noise_in_dist
        cell_id = self.cell_ids[cell_id]
        cell_id[dist > max_dist] = -1
        pixel_map = np.reshape(cell_id, x_coord.shape)
        
        #remove cells without pixels
        uniq_cells = np.unique(pixel_map.ravel())
        cells_to_remove = []
        for i in self.cell_ids:
            if i not in uniq_cells:
                cells_to_remove.append(i)
        
        #remove cells with less than a certain pixel number
        counted = Counter(pixel_map.ravel())
        for c in counted:
            if counted[c] < min_pix_count:
                pixel_map[pixel_map == c] = -1
                cells_to_remove.append(c)

        self.cell_ids = np.delete(self.cell_ids, cells_to_remove)
        self.cell_centers.drop(index=cells_to_remove, inplace=True)
        self.true_map = pixel_map
        
    def plot_true(self, cmap = 'nipy_spectral',ax=None, alpha = 1, z_stack = 5):
        '''
            Plots the cell ID grid in a single z_stack value
            --------------------
            parameters:
                cmap: the color map from matplot lib to use (default: nipy_spectral)
                ax: the matplotlib cell to plot in (default: None)
                alpha: measure of the transparency of the image (default: 1)
                z_stack: number for the z slice to plot in (default: 0)
        '''    
        if self.true_map is None:
            self.assign_pixels_to_cells()
            
        true_map = self.true_map.copy()
        new_ids = np.unique(self.true_map.ravel())
        new_ids = np.delete(new_ids, 0)
        np.random.shuffle(new_ids)
        
        counter = 0
        for i in new_ids:
            true_map[self.true_map == i] = counter
            counter += 1
        
        if ax is None:
            plt.imshow(true_map[:,:,z_stack], cmap=cmap, alpha = alpha)
        else:
            ax.imshow(true_map[:,:,z_stack], cmap=cmap, alpha = alpha)
    

    def plot_celltypes(self, cmap = 'nipy_spectral',
                  ax = None, alpha = 1, z_stack = 0):
        '''
            Plots the celltype grid in a single z_stack value
            --------------------
            parameters:
                cmap: the color map from matplot lib to use (default: nipy_spectral)
                ax: the matplotlib cell to plot in (default: None)
                alpha: measure of the transparency of the image (default: 1)
                z_stack: number for the z slice to plot in (default: 0)
            --------------------
            adds:
                self.cellt_map: Adds the 3d cell type map -1 is no cell, cell type number
                    is based on the order of self.celltypes
        '''   
        cellt = self.classified_celltypes
        colors = []
        celltype_colors = {}
        for i, cell in enumerate(np.unique(cellt)):
            celltype_colors[cell] = i
        for cell in cellt:
            colors.append(celltype_colors[cell])
        colors = np.array(colors)
        t_map = self.true_map.ravel().copy()
        for i, cell_id in enumerate(np.unique(t_map)):
            if cell_id != -1:
                t_map[t_map == cell_id] = i - 1
        cellt_vec = colors[t_map]
        cellt_vec[t_map == -1] = -1
        cellt_mat = cellt_vec.reshape(self.true_map.shape)
        self.cellt_map = cellt_mat
        
        
        # create legend
        leg = np.unique(cellt)
        cmap_converter = plt.cm.get_cmap(cmap, len(leg)+1)
        leg_patch = []
        for i in range(len(leg)):
            leg_patch.append(mpatches.Patch(color=cmap_converter(i+1)[:3],
                             label=leg[i]))
        if ax is None:
            plt.imshow(cellt_mat[:,:,0], cmap = cmap, alpha = alpha)
            plt.legend(handles = leg_patch,bbox_to_anchor=(2,1))
        else:
            ax.imshow(cellt_mat[:,:,0], cmap = cmap, alpha = alpha)
            ax.legend(handles = leg_patch,bbox_to_anchor=(2,1))
        
    
    def generate_nuclei_centers(self, n_pix_per_nuc = 9, dtype='int32'):
        '''
            Generates the nuclei randomly within a cell
            --------------------
            parameters:
                n_pix_per_nuc: number of pixels in a nucleus (default: 9)
                dtype: data type for the nuclei tensor (default: int16)
            --------------------
            adds:
                self.nuclei: 3d tensor of nuclei, -1 is no nucleus, number is according
                    to the cell ID
        '''
        if self.cell_centers is None:
            self.generate_cell_centers()

        nuclei = np.zeros(self.true_map.shape,
                         dtype=dtype)
        nuclei -= 1
        
        check_nuclei_surroundings = KNeighborsClassifier(n_neighbors = 27)

        non_zero_cell_locs = np.where(self.true_map != -1)
        cell_ids = self.true_map.ravel()
        xs = non_zero_cell_locs[0]
        ys = non_zero_cell_locs[1]
        zs = non_zero_cell_locs[2]
        cell_ids = cell_ids[cell_ids != -1]
        check_nuclei_surroundings.fit(np.vstack((xs,ys,zs)).T, cell_ids)

        for i in np.unique(self.cell_ids):
            if i != -1:
                cell_coords = np.where(self.true_map == i)
                rand_index = randint(0,len(cell_coords[0])-1)
                xs = cell_coords[0]
                ys = cell_coords[1]
                zs = cell_coords[2]

                locs_as_mat = np.vstack((xs, ys, zs)).T
                clf_seed = KNeighborsClassifier(n_neighbors=min(n_pix_per_nuc,
                                                               locs_as_mat.shape[0]))
                nuclei_seed = locs_as_mat[rand_index,:]

                clf_seed.fit(locs_as_mat, np.arange(locs_as_mat.shape[0]))
                nuc_pix_locs = clf_seed.kneighbors([nuclei_seed])[1][0]
                nuc_pix = locs_as_mat[nuc_pix_locs,:]

                non_same_celltype = cell_ids[check_nuclei_surroundings.kneighbors(nuc_pix)[1]]
                #remove nuc pixels that are on the border
                nuc_pix = nuc_pix[np.sum(non_same_celltype == i,axis=1) == 27,:]
                if nuc_pix.shape[0] > 0:
                    nuclei[nuc_pix[:,0],
                          nuc_pix[:,1],
                          nuc_pix[:,2]] = i
                else:
                    nuclei[nuclei_seed[0],
                          nuclei_seed[1],
                          nuclei_seed[2]] = i
                #print(np.unique(nuclei))

        self.nuclei = nuclei
    
    def compute_covariance(self, counts,
                            celltypes,
                            find_celltype_props = True):
        '''
            Computes and adds the DGE  matrices to the object. It should be in celltypes x genes
            Also computes cell type proportions
            --------------------
            parameters:
                counts: count matrix of single cell data cells x genes
                celltypes: vector of celltypes for each cell in sc matrix
                find_celltype_props: indicating whether or not to use the celltype
                    proportions from the single cell data (default: True)

            --------------------
            adds:
                self.genes: vector of the genes
                self.celltypes: vector of the celltypes
                self.ct_means: matrix with mean gene expression of each gene for each celltype
                self.ct_stds: matrix with stdev of gene expression of each gene for each celltype
                self.ct_covs: covariance matrix for each celltype
                self.celltype_props: if indicated, adds the celltype proportions from 
                    the single cell data
        '''
        self.genes = counts.columns.to_numpy()
        self.celltypes = np.unique(celltypes)
        
        celltype_means = np.zeros((len(np.unique(celltypes)),counts.shape[1]))
        celltype_stds = np.zeros((len(np.unique(celltypes)),counts.shape[1]))
        cov_mats = np.zeros((len(np.unique(celltypes)),counts.shape[1],counts.shape[1]))
        for i,cell in enumerate(np.unique(celltypes)):
            loc = np.where(cell == celltypes)[0]
            subset_cells = counts.iloc[loc,:]
            celltype_means[i,:] = np.mean(subset_cells,axis=0)
            celltype_stds[i,:] = np.std(subset_cells,axis=0)
            
            x = np.cov(subset_cells.T)
            min_eig = np.min(np.real(np.linalg.eigvals(x)))
            if min_eig < 0:
                x -= 100*min_eig * np.eye(*x.shape)
            cov_mats[i,:,:] = x

           

        self.ct_means = celltype_means
        celltype_stds[celltype_stds == 0] = 1
        self.ct_stds = celltype_stds
        self.ct_covs = cov_mats

        
        if find_celltype_props:
            celltype_props = []
            for i, cell in enumerate(self.celltypes):
                celltype_locs = np.where(celltypes == cell)[0]
                celltype_props.append(len(celltype_locs)/len(celltypes))
            self.celltype_props = np.array(celltype_props)
        
    def classify_celltypes(self,subtype = False, ct_list = None, st_list = None):
        '''
            Adds celltypes to the cells based on the cell type proportions
            --------------------
            parameters:
                ct_list: list of the celltype annotation for the subt
            --------------------
            adds:
                self.celltype_props: if there aren't available cell type proportions 
                    sets them uniform
                self.classified_celltypes: vector of the cell type for each cell
        '''
        if self.celltypes is None:
            print('No Celltypes Available')
            return
        if self.celltype_props is None:
            print('No celltype proportions available. Assuming uniform ')
            self.celltype_props = [1/len(self.celltypes) for i in range(len(self.celltypes))]
        cell_probs = np.tile(self.celltype_props,(len(self.cell_ids),1))
        rand_unif = np.random.rand(len(self.cell_ids))
        class_celltype_index = (cell_probs.cumsum(1) > np.random.rand(len(self.cell_ids))[:,None]).argmax(1)
        
        self.classified_celltypes = self.celltypes[class_celltype_index]
        if self.subtype:
            ct_map = {}
            for i,ct in enumerate(st_list):
                if ct not in ct_map:
                    ct_map[ct] = ct_list[i]
            self.classified_celltypes_lowres = np.array([ct_map[i] for i in self.classified_celltypes])
        
    def generate_merfish_dge(self,dge_scaling_factor = 1e1):
        '''
            Generates the merfish dge for the cells, based on their cell type and the dge from
                single cell data
            --------------------
            parameters:
                dge_scaling_factor: multiplies the dge matrix by this number
                    sometimes the single cell computed matrix values are too small due to sparsity
                    this increases the values for later use in generating spots (default: 1e1)
            --------------------     
            adds:
                merfish_dge: cells x genes dge matrix for the merifsh cells
        '''
        merfish_dge = np.zeros((len(self.classified_celltypes),
                               len(self.genes)))
        uniq_celltypes = np.unique(self.classified_celltypes)
        for i, cell in enumerate(self.cell_ids):
            sum_of_count = 0
            cellt = self.classified_celltypes[i]
            cellt_ind = np.where(uniq_celltypes == cellt)[0][0]

            counts = np.random.multivariate_normal(self.ct_means[cellt_ind,:],
                                                  self.ct_covs[cellt_ind,:,:])
            counts *= self.ct_stds[cellt_ind,:]
            counts += self.ct_means[cellt_ind,:]
            merfish_dge[i,:] = counts
        
        merfish_dge *= dge_scaling_factor
        merfish_dge[merfish_dge < 0] = 0
        merfish_dge = np.round(merfish_dge)
        merfish_dge = pd.DataFrame(merfish_dge)
        merfish_dge.columns = self.genes
        merfish_dge.index = self.cell_ids
        self.merfish_dge = merfish_dge
        #self.merfish_dge = np.round(self.merfish_dge)
        #self.merfish_dge[self.merfish_dge <= 0] = 0

    #dist_from_nuc_scale 0 is uniform dist, 1 is all right next to nuc
    def place_transcripts(self, dist_from_nuc_scale = 0):
        '''
            Places the spots within the cell
            --------------------
            parameters:
                dist_from_nuc_scale: indicates the uniformity of spots within a cell,
                    0 is uniform distribution, as it increases, the spots cluster around the nucleus
            --------------------
            adds:
                self.spots: spot calls matrix cells x 4 (gene, x, y, z)
                self.nuc_df: nuclei df matrix cells x 4 (id, x, y, z)
        '''
        t_map = self.true_map
        nuc_map = self.nuclei
        
        if dist_from_nuc_scale <= 0:
            dist_from_nuc_scale = 1e-3
        
        n_exp_pdf = len(self.cell_ids)*10
        
        #psuedo exponential distribution
        exp_pdf = np.exp(dist_from_nuc_scale*np.random.rand(n_exp_pdf))
        
        exp_pdf = np.sort(exp_pdf)
        exp_pdf /= np.sum(exp_pdf)
        
        exp_cdf = np.array(np.cumsum(exp_pdf))[::-1]
        
        n_nuc_pix = np.sum(nuc_map != -1)
        nuc_df = np.zeros((n_nuc_pix, 4))
        
        #spots = pd.DataFrame(columns = ['gene','x','y','z'])
        
        spots = np.zeros((int(np.sum(self.merfish_dge.to_numpy().ravel())),3),
                        dtype=np.uint32)
        
        gene_vec = []
        spots_mat_iter = 0 
        nuc_iter = 0
        tot_spot = 0
        for i in np.unique(nuc_map.ravel()):
            if i != -1:
                nuc_loc = np.array(np.where(nuc_map == i)).T
                nuc_df[nuc_iter:nuc_loc.shape[0]+nuc_iter,0] = i
                nuc_df[nuc_iter:nuc_loc.shape[0]+nuc_iter,1:] = nuc_loc
                nuc_iter += nuc_loc.shape[0]
                
                whole_cell_loc = np.array(np.where(t_map == i)).T
                
                nuc_mid = [np.mean(nuc_loc,axis=0)]
                clf_cell = KNeighborsClassifier(n_neighbors=whole_cell_loc.shape[0]).fit(whole_cell_loc,
                                                                                      np.arange(whole_cell_loc.shape[0]))
                cell_count = self.merfish_dge.loc[i,:]
                indices = clf_cell.kneighbors(nuc_mid)[1][0]
                n_counts = int(np.sum(cell_count))
                n_locs = whole_cell_loc.shape[0]    
                
                random_count_locs = exp_cdf[np.array(np.round(np.random.rand(n_counts)*(n_exp_pdf-1)),dtype=int)]             
                random_count_locs *= n_locs-1
                random_count_locs = np.array(np.round(random_count_locs),dtype=int)

                spot_ind = indices[random_count_locs]
                spot_locs = whole_cell_loc[spot_ind,:]

                non_zero_genes = np.where(cell_count != 0)[0]
                spot_iter = 0
                for j in non_zero_genes:
                    gene = self.genes[j]
                    n_spots = int(cell_count[j])
                    tot_spot += n_spots
                    for k in range(n_spots):
                        pix_loc = spot_locs[spot_iter,:]
                        row = [pix_loc[0], pix_loc[1], pix_loc[2]]
                        gene_vec.append(gene)
                        spots[spots_mat_iter,:] = row
                        spot_iter += 1
                        spots_mat_iter += 1
                        
        spots = spots[0:len(gene_vec),:]
        spots = pd.DataFrame(spots)
        spots.columns = ['x','y','z']
        gene_vec = np.array(gene_vec)
        spots['gene'] = gene_vec

        nuc_df = pd.DataFrame(nuc_df)
        nuc_df.columns = ['id','x','y','z']
        
        self.spots = spots
        self.nuc_df = nuc_df
    
    def place_transcripts_at_corners(self):
        '''
            Places a spot from a random gene at each corner to make sure 
            The pixel tensor size ends up the same as the simulated
        '''
        y_max, x_max, z_max = self.grid_shape
        y_max *= self.pix_per_micron
        x_max *= self.pix_per_micron
        z_max *= self.pix_per_micron
        
        y_min = 0; x_min = 0; z_min = 0;
        for x in [x_min, x_max-1]:
            for y in [y_min, y_max-1]:
                for z in [z_min, z_max-1]:
                    rand_gene = self.genes[int(np.random.rand(1)*len(self.genes))]
                    self.spots.loc[self.spots.shape[0],:] = [x, y, z, rand_gene]
                    
    def add_noise(self, avg_spots_per_cell, percent_empty_to_use):
        '''
            Adds noise to empty space 
            --------------------
            parameters:
                avg_spots_per_cell: the average number of extra spots in each empty cell chosen
                percent_empty_to_use: percentage of the empty squares to put noise in
        '''
        empty_slots = np.where(mdg.true_map)
        nempty = len(empty_slots[0])
        noise_slots = np.random.choice(np.arange(nempty),
                                      size = int(nempty * percent_empty_to_use))
        num_spots = np.round(np.random.rand(len(noise_slots)) * 2 * avg_spots_per_cell)
        num_genes_to_add = int(np.sum(num_spots))
        rand_genes = self.genes[(np.random.rand(num_genes_to_add)*len(self.genes)).astype(int)]
        noise_slots = noise_slots[num_spots != 0]
        num_spots = num_spots[num_spots != 0]
        new_spots = np.zeros(((self.spots.shape[0]+len(rand_genes)),4))
        new_spots[0:self.spots.shape[0],0:3] = self.spots.loc[:,['x','y','z']]
        new_genes_counter = self.spots.shape[0]

        for i in range(len(num_spots)):
            x = empty_slots[0][noise_slots[i]]
            y = empty_slots[1][noise_slots[i]]
            z = empty_slots[2][noise_slots[i]]
            for j in np.arange(num_spots[i]):
                new_spots[new_genes_counter,0:3] = [x,y,z]
                new_genes_counter += 1
                
        new_spots = pd.DataFrame(new_spots)
        new_spots.columns = self.spots.columns
        new_spots.gene = np.concatenate((self.spots.gene.to_numpy(),
                                        rand_genes))
        self.spots = new_spots
        
    def merge_cells(self, n_iter = 1):
        '''
            Merges cells to create non-circular shapes
            --------------------
            parameters:
                n_iter: number of times to merge cells. (Default: 1)
        '''
        for n in range(n_iter):
            coords = []
            ids = []
            for i in range(self.true_map.shape[0]):
                for j in range(self.true_map.shape[1]):
                    for k in range(self.true_map.shape[2]):
                        ids.append(self.true_map[i,j,k])
                        coords.append([i,j,k])
            coords = np.array(coords)
            ids = np.array(ids)
            clf_nei_cells = KNeighborsClassifier(27).fit(coords,ids)
            pred = ids[clf_nei_cells.kneighbors(coords)[1]]
            combine_map = {}
            for i in range(pred.shape[0]):
                if pred[i,0] != -1:
                    same_locs = np.where((pred[i,1:] != pred[i,0])&
                                        (pred[i,1:] != -1))[0]

                    if len(same_locs) > 5:
                        #plus one because the where was run on pred[i,1:] 
                        combine_map[pred[i,0]] = pred[i,(same_locs[0]+1)]
            flipped_before = []
            for flip in combine_map:
                if ((flip not in flipped_before) &
                    (combine_map[flip] not in flipped_before)):
                    self.true_map[self.true_map == flip] = combine_map[flip]
                    self.cell_ids[self.cell_ids == flip] = combine_map[flip]
                    flipped_before.append(flip)
                    flipped_before.append(combine_map[flip])
        self.cell_ids = np.unique(self.cell_ids)
        


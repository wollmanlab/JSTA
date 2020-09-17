
def reclassify_squares(pix, pixl_true, surround_count, 
                       same_cts,
                       cell_matrix, nuc, nuc_clf,
                       cell_assign, empty_pix, sc_ref, sc_ref_celltypes, all_genes, 
                       locs, genes_to_use_prediction,
                       n_celltype, npcs,
                      pct_train,  border_other_threshold,
                       border_same_threshold,
                       outer_max, inner_max,most_inner_max,
                      dist_threshold, dist_scaling, anneal_param,
                       flip_thresh, final_prune_prob,
                      clf_cell):
    '''
    Method to flip pixels from one cell to another or to no cell assignment to improve cell segmentation
    High level description: 
        a) Classify voronoi defined cells to a cell type according to a soft assignment.
        b)Train on a subset of the pixels to build a pixel level
            classifier for determining the celltype identity 
            of pixels in PCA space of the kde of genes. 
        c) Flip border pixels acording to their predictions using the model in b.
        Keep switching between a, b, and c flipping pixels and retraining the models
    ---------------------
    parameters:
        pix: 4d tensor. 4th dimension is gene expression (kde) vector of each pixel
        pixl_true: 4d tensor. 4th dimension is gene expression (counts) vector of each pixel
        surround_count: 4d tensor. 4th dimension is the number of surroundings of each cell type
        cell_matrix: voronoi segmented digital gene expression matrix cells x genes
        nuc: data frame with x,y,z coordinates of nuclei pixels with nuclei IDs
        nuc_clf: pre-trained knn classifier on nuclei pixels
        cell_assign: 3d tensor. each pixel has it's current nuclei classification or none
        empty_pix: 3d binary tensor. 1 indicates not empty 0 indicates empty
        sc_ref: scRNAseq reference matrix cells x genes
        sc_ref_celltypes: vector of cell types for each scRNAseq cell
        all_genes: genes in merfish data
        locs: 4d tensor. 4th dimension is an x,y,z vector of the coordinates of the center of that pixel
        genes_to_use_prediction: overlapping genes between merfish and scRNAseq
        n_celltype: number of cell types in scRNAseq data
        npcs: number of dimensions to use in PCA
        pct_train: percentage of pixels to train on for pixel classifier
        border_other_threshold: how many pixels need to belong to the other cell to flip to that cell
        outer_max: numbegr of iterations of the outer loop (cell classification (a))
        inner_max: number of iterations of the inner loop (pixel training (b))
        most_inner_max: number of iterations of the flipping pixels loop (pixel flip (c))
        dist_threshold: distance from edge of nucleus to ensure a cell belongs to that nucleus
        dist_scaling: amount of decay of probabilities for flipping as you move away from a nucleus of
            interest. The probability decreases by half every dist_threshold*dist_scaling
        annealing_param: Parameter to decrease the probabalistic component of pixel and cell classification.
            Every iteration the highest probability is multiplied by 1+annealing_param*n_iteration
        clf_cell:  neural network based cell type predictor
    return:
        cell_assign: 3d tensor giving the nuclei assignment of each pixel 
    '''
    #copy of original assignment for later use
    copy_cell_assign = cell_assign.copy()

    #gets the genes we need for cell type prediction
    gene_subset_indices = []
    for i in all_genes:
        if i in genes_to_use_prediction:
            gene_subset_indices.append(True)
        else:
            gene_subset_indices.append(False)
    

    nuc_clf = neighbors.NearestNeighbors(n_neighbors=10, n_jobs = 4).fit(nuclei.loc[:,['x','y','z']],
                                                                       nuclei.id)

    #perc_right, perc_pred_right = compute_metrics(cell_assign, t_map)
    
    #perc_right_celltype_list = []
    #perc_right_celltype_predicted_list = []
    #perc_right_list = []
    #perc_right_pred_list = []
    #perc_right_list.append(perc_right)
    #perc_right_pred_list.append(perc_pred_right)

    pix_shape = pix.shape
    x_max, y_max, z_max = pix_shape[0]-1, pix_shape[1]-1, pix_shape[2]-1
    
    print('computing pca on pixels')
    tic = time()
    #perform PCA on the pixels
    cp_grid = pixel_pca(pix, npcs)

    toc = time()
    print('time to compute pca ',toc-tic)

    num_iterations_outer = 0
    np.seterr(invalid='raise')
    square_param_diff_vec = []
    #marker_gene_overlap = np.isin(all_genes,m_genes)
    prediction_mean = []
    p_weight = None
    p_mean = None
    percent_flipped = []
    logi_param_diff = []

    #center and scale sc ref
    #sc_ref.loc[:,:] = scale(sc_ref, axis=0)
    
    overlapping_genes_for_merfish_map = np.isin(all_genes,
                                                sc_ref.columns)

    n_combined_cells = cell_matrix.shape[0]+sc_ref.shape[0]
    combined_cells = np.zeros((n_combined_cells,
                              np.sum(overlapping_genes_for_merfish_map)))
    
    clf_log = pixel_nn_classifier(sc_ref,
                                  sc_ref_celltypes,
                                  3,
                                 1e-3)

    n_iterations = 0
    n_changed = []
    n_changed_overall = []
    
    while(num_iterations_outer < outer_max):
        #print('outer iterations:',num_iterations_outer)
        cells_matrix = get_matrix_of_cells(pixl_true,cell_assign, nuc).to_numpy()
        
        print(np.mean(np.sum(cells_matrix,axis=1)))
        print(np.min(np.sum(cells_matrix,axis=1)))
        non_empty_cell_locs = np.where(np.sum(cells_matrix,axis=1) > 100)[0]
        print(cells_matrix.shape)
        print(len(non_empty_cell_locs))
        cells_matrix = scale(cells_matrix,axis=1)
        cells_matrix = scale(cells_matrix,axis=0)
        tic = time()

        # combines the mer and sc_ref matrices and builds a classifier for celltype
        # based on sc_ref
        print('finding celltypes')
        cells_probs = clf_cell.predict(cells_matrix)

        toc = time()
        print('time to get celltypes',toc-tic)
        #if n_iterations == 0:
            #np.savetxt('real_results/hyp_preop/classified_em/celltype_probs_presegment.'+file_index+'.tsv',
            #      cells_probs, delimiter='\t')
        prediction_mean.append(np.mean(np.max(cells_probs,axis=1)))
        print(prediction_mean)
        #adding multiply the max prob by 1+n_iteration*annealing_param
        max_pred = np.argmax(cells_probs,axis=1)

        cells_probs[np.arange(len(max_pred)),
                   max_pred] *= 1+n_iterations*anneal_param
        
        cells_probs /= np.sum(cells_probs,axis=1,keepdims=True)
        print(np.mean(np.max(cells_probs,axis=1)))
        
        
        #print('PREDICTION MEAN ',prediction_mean)
        #print(prediction_mean)
        #get the identity of the predicted cell types
        groupings = (cells_probs.cumsum(1) > np.random.rand(cells_probs.shape[0])[:,None]).argmax(1)
        #groupings = np.argmax(cells_probs,axis=1)

        #groupings = t_cell

        #if num_iterations_outer == 0:
            #perc_right, perc_pred_right = compute_cellspec_metrics(cell_assign, t_map, groupings, t_cell)
            #perc_right_celltype_list.append(perc_right)
            #perc_right_celltype_predicted_list.append(perc_pred_right)


        
        #print(Counter(groupings))


        last_param = None
        #clf_log = LogisticRegression(random_state=0,max_iter=5000,#warm_start=True,
        #                        n_jobs=4,penalty='l2')
        
        toc = time()
        #print('time to find cell types ',toc-tic)

        num_iterations_inner = 0
        past_square_param = None
        while(num_iterations_inner < inner_max):
            #print('inner iterations:',num_iterations_inner)
            flat_assign = np.ravel(cell_assign)
            group_labels = groupings[flat_assign]
            
            #the empty cell type will be index of number of celltypes
            group_labels[flat_assign == -1] = n_celltype

            #random selection of pixels to use for training the model
            subset_indices = np.random.choice(np.arange(0,
                                                    flat_assign.shape[0],dtype=int),
                                          size=int(pct_train*len(group_labels)))
            tic = time()
            merged_pix_info = cp_grid
            print(merged_pix_info.shape)
            merged_pix_shape = merged_pix_info.shape
            merged_pix_reshaped = np.reshape(merged_pix_info,(merged_pix_shape[0]*merged_pix_shape[1]*merged_pix_shape[2],merged_pix_shape[3]))
            group_labels_mat = np.reshape(group_labels, (merged_pix_shape[0],merged_pix_shape[1],merged_pix_shape[2]))
            
            sub_merged_pix = merged_pix_reshaped[subset_indices,:]
            sub_group_labels = group_labels[subset_indices]

            #remove non cell pixels from training
            sub_merged_pix = sub_merged_pix[sub_group_labels != n_celltype,:]
            sub_group_labels = sub_group_labels[sub_group_labels != n_celltype]
            
            
            #train a model to see what a pixel of a certain kind looks like
            #clf_log.fit(sub_merged_pix, sub_group_labels)
            if ((num_iterations_inner == 0)):
                clf_log = train_nn_classifier(sub_merged_pix, sub_group_labels,clf_log,
                                             25,[1e-3,1e-4])
            else:
                clf_log = train_nn_classifier(sub_merged_pix, sub_group_labels,clf_log,
                                             15,[1e-4])

            #square_param = clf_log.coef_
            
            #print(square_param)
            #if last_param is None:
            #    last_param = square_param
            #else:
            #    logi_param_diff.append(np.linalg.norm(last_param - square_param))
            #    last_param=square_param
            
            toc = time()
            print('time to train ',len(subset_indices),'samples ',toc-tic)
            num_iterations_most_inner = 0
            while(num_iterations_most_inner < most_inner_max):
                #print(num_iterations_most_inner)
                #print('inner most iterations:',num_iterations_mamost_inner)
                #merged_pix_info = cp_grid
                #merged_pix_shape = merged_pix_info.shape
                #merged_pix_reshaped = np.reshape(merged_pix_info,(merged_pix_shape[0]*merged_pix_shape[1]*merged_pix_shape[2],merged_pix_shape[3]))
                #group_labels_mat = np.reshape(group_labels, (merged_pix_shape[0],merged_pix_shape[1],merged_pix_shape[2]))

                #find border indices where they are not empty, and have more neighbors than the border
                #threshold
                border_indices = np.where(#(empty_pix == 1)&
                                         (surround_count >= border_other_threshold)&
                                         (same_cts >= border_same_threshold))

                border_indices_mat = np.stack([border_indices[0],border_indices[1],border_indices[2]],axis=1)
                #print('number of border pixels:',border_indices_mat.shape)
                tic = time()
                #predict the probability a border pixel is from each class
                predictions = clf_log.predict(merged_pix_info[border_indices[0],border_indices[1],border_indices[2],:])            
                
                predictions[predictions <= flip_thresh] = 0
                
                ngene = merged_pix_info.shape[3]
                ms = merged_pix_info.shape
                npix = ms[0]*ms[1]*ms[2]
                #pred_all = clf_log.predict_proba(merged_pix_info[:,:,:,:].reshape((npix,ngene)))

                #adding multiply the max prob by 1+n_iteration*annealing_param
                max_pred = np.argmax(predictions,axis=1)
                
                
                predictions[np.arange(len(max_pred)),
                           max_pred] *= 1+n_iterations*anneal_param

                if predictions.shape[1] < (n_celltype):
                    #diff = np.setdiff1d(np.unique(groupings),np.unique(group_labels[subset_indices]))
                    diff = np.setdiff1d(np.arange(n_celltype+1),np.unique(group_labels[subset_indices]))
                    diff = np.sort(diff)
                    for missing in diff:
                        predictions = np.insert(predictions, missing, np.repeat(0,predictions.shape[0]),axis = 1)
                    print('MISSSING ROW INSERTED!')


                toc = time()
                print('time to predict ',len(border_indices[0]),'samples',toc-tic)
                #get the number of each cell type pixel surrounding each border pixel

                
                #some house keeping for the next step by padding arrays
                border_indices_mat += 1
                cell_assign = np.pad(cell_assign, (1), 'constant',constant_values=(-2))

                #nuc_dens = np.pad(nuc_dens, (1), 'constant',constant_values=(-2))
                locs = np.pad(locs, ((1,1),(1,1),(1,1),(0,0)), 'constant',constant_values=(-2))

                bord_x, bord_y, bord_z = border_indices_mat[:,0], border_indices_mat[:,1], border_indices_mat[:,2]
                surroundings = np.zeros((border_indices_mat.shape[0]),dtype=int)
                group_key = np.zeros((border_indices_mat.shape[0]),dtype=int)
                predic_num = np.zeros_like(predictions)
                predic_probs = np.zeros((predictions.shape[0],predictions.shape[1]))
                tic = time()
                
                pixels_to_nuc_dist_vec, pixels_to_nuclei_vec = nuc_clf.kneighbors(locs[bord_x,
                                                                                      bord_y,
                                                                                      bord_z,:])


                pixels_to_nuclei_vec = nuc.id.to_numpy()[pixels_to_nuclei_vec]
                toc = time()
                print('time to predict nuclei distance:',toc-tic)

                tic = time()
                
                #this creates a matrix where each row is the surrounding cell ids for that border pixel
                surroundings = np.zeros((len(bord_x),27))
                sub_counter = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        for k in range(-1,2):
                            surroundings[:,sub_counter] = cell_assign[bord_x+i,
                                                                    bord_y+j,
                                                                    bord_z+k].copy()
                            sub_counter += 1
                
                tic = time()
                
                input_surr = (surroundings.__array_interface__['data'][0]+np.arange(surroundings.shape[0])*surroundings.strides[0]).astype(np.uintp)
                input_nuc = (pixels_to_nuclei_vec.__array_interface__['data'][0]+np.arange(pixels_to_nuclei_vec.shape[0])*pixels_to_nuclei_vec.strides[0]).astype(np.uintp)
                input_dist = (pixels_to_nuc_dist_vec.__array_interface__['data'][0]+np.arange(pixels_to_nuc_dist_vec.shape[0])*pixels_to_nuc_dist_vec.strides[0]).astype(np.uintp)
                dist_mat = np.zeros((surroundings.shape[0],27),dtype=np.float64)
                input_dist_mat = (dist_mat.__array_interface__['data'][0]+np.arange(dist_mat.shape[0])*dist_mat.strides[0]).astype(np.uintp)
                #written in C
                get_d(input_surr,
                    input_nuc,
                    input_dist,
                    ctypes.c_int(surroundings.shape[0]),
                    ctypes.c_int(pixels_to_nuclei_vec.shape[1]),
                   input_dist_mat)
                
                toc = time()
                #print('time to get dist mat:',toc-tic)
                
                tic = time()
                for sub_counter in range(27):
                    surro = surroundings[:,sub_counter].copy().astype(int)

                    num_surroundings = len(surro)
                    # get dist vec based on the first occurence of the surroundings
                    # in the nearest neighbors vec from pix to nuclei
                    dist_vec = dist_mat[:,sub_counter]
                    
                    reduced_dist = dist_vec - dist_threshold
                    reduced_dist[reduced_dist <= 0] = 1e-3

                    s = (dist_scaling*dist_threshold)/2
                    scale_vec = np.divide(s,reduced_dist)
                    scale_vec[dist_vec < dist_threshold] = 10
                    scaled_final = np.minimum(10,scale_vec)
                    
                    scaled_final[np.where(surro == -1)] = 1
                    scaled_final[np.where(surro == -2)] = 1

                    group_key = groupings[surro].copy()
                    group_key[np.where(surro == -1)] = -1
                    group_key[np.where(surro == -2)] = -2
                    non_neg = np.where(group_key >= 0)
                    predic_num[non_neg,group_key[non_neg]] += 1
                    predic_probs[non_neg,group_key[non_neg]] += scaled_final[non_neg]

                predic_num[np.where(predic_num == 0)] = 1
                predic_probs = np.divide(predic_probs, predic_num)
                #to make sure our null data type keeps prob the same
                #predic_probs[:,predic_probs.shape[1]-1] = 1
                
                #predic_probs[predic_probs < empty_thresh] = 0
                
                predictions = np.multiply(predictions, predic_probs)

                #predictions[predictions > 0] = 1

                pred_sum = np.sum(predictions,axis=1,keepdims=1)
                pred_sum[pred_sum == 0] = 1
                predictions /= pred_sum

                #print(np.sum(np.sum(predictions, axis=1) == 0))
                #return predictions
                toc = time()
                #print('time to get distance:', toc-tic)
                
                #predictions[np.arange(0,predictions.shape[0]), same_group_mask] = 0
                #border_indices_mat = border_indices_mat[np.where(np.sum(predictions,axis=1)>0)[0],:]
                number_of_border_pixels = border_indices_mat.shape[0]
                #predictions = predictions[np.where(np.sum(predictions,axis=1)>0)[0],:]
                #return predictions
                locs_to_flip = locs[border_indices_mat[:,0], border_indices_mat[:,1], border_indices_mat[:,2],:]
                #remove 0 pad from nuc_dens
                #ns = nuc_dens.shape
                #nuc_dens = nuc_dens[1:ns[0]-1,1:ns[1]-1,1:ns[2]-1]
                
                #STOPPED EVALUATING HERE
                
                #remove 0 pad from locs
                ls = locs.shape
                locs = locs[1:ls[0]-1,1:ls[1]-1,1:ls[2]-1,:]

                flips_arg_bool = (predictions.cumsum(1) > np.random.rand(predictions.shape[0])[:,None])
                
                flips_arg = flips_arg_bool.argmax(1)
                
                
                #set flip args that have prediction prob 0 to n_celltype
                flips_arg[np.sum(flips_arg_bool, axis=1) == 0] = n_celltype
                flips_arg[np.sum(predictions,axis=1) == 0] = n_celltype
                
                cells_to_flip = flips_arg

                borders_to_flip = border_indices_mat

                tic = time()


                bord_x, bord_y, bord_z = borders_to_flip[:,0], borders_to_flip[:,1], borders_to_flip[:,2]
                old_id = cell_assign[bord_x,
                                    bord_y,
                                    bord_z].copy()

                surroundings = np.zeros((border_indices_mat.shape[0],3**border_indices_mat.shape[1]),dtype=int)
                group_key = np.zeros((border_indices_mat.shape[0],3**border_indices_mat.shape[1]),dtype=int)
                matching_regions = np.zeros((border_indices_mat.shape[0],3**border_indices_mat.shape[1]),dtype=np.float32)
                changed_surround_count = np.zeros((border_indices_mat.shape[0],3**border_indices_mat.shape[1]),dtype=int)
                loc_counter = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        for k in range(-1,2):
                            surroundings[:,loc_counter] = cell_assign[bord_x+i,
                                                                        bord_y+j,
                                                                        bord_z+k].copy()
                            
                            group_key[:,loc_counter] = groupings[surroundings[:,loc_counter]].copy()
                            group_key[:,loc_counter][np.where(surroundings[:,loc_counter] == -1)] = n_celltype
                            group_key[:,loc_counter][np.where(surroundings[:,loc_counter] == -2)] = -2

                            matching_regions[:,loc_counter] = group_key[:,loc_counter] == cells_to_flip
                            loc_counter += 1
                            
                match_dist = matching_regions * dist_mat
                match_dist[match_dist == 0] = 1e3
                new_cell_loc = np.argmin(match_dist,axis=1)
                                
                #return surroundings, matching_regions, dist_mat
                #matching_regions /= np.maximum(matching_regions.sum(axis=1,keepdims=1),1)
                #new_cell_loc = (matching_regions.cumsum(1) > np.random.rand(matching_regions.shape[0])[:,None]).argmax(1)

                new_id = surroundings[np.arange(0,surroundings.shape[0]),new_cell_loc].copy()
                new_id[np.sum(matching_regions,axis=1) == 0] = -1
                new_id[np.sum(flips_arg_bool, axis=1) == 0] = -1
                new_id[new_id == -2] = -1
                
                n_changed_this_round = len(new_id) - np.sum(np.equal(old_id,new_id))
                n_changed.append(n_changed_this_round)
                n_changed_overall.append(n_changed_this_round)
                if len(n_changed) > 10:
                    n_changed = n_changed[1:]
                #print(n_changed)

                cell_assign[bord_x,bord_y,bord_z] = new_id

                
                same_and_different = surroundings == new_id.reshape(surroundings.shape[0],1)
                different_count = np.sum(same_and_different == 0,axis=1)
                borders_to_flip -= 1
                
                #remove 0 pad from cell assignment
                cs = cell_assign.shape
                cell_assign = cell_assign[1:cs[0]-1,1:cs[1]-1,1:cs[2]-1]
                cell_assign[copy_cell_assign != -1] = copy_cell_assign[copy_cell_assign != -1].copy()                
                #evaluate metrics
                #perc_right, perc_pred_right = compute_metrics(cell_assign, t_map)
                #perc_right, perc_pred_right = compute_cellspec_metrics(cell_assign, t_map, groupings, t_cell)
                #perc_right_list.append(perc_right)
                #perc_right_pred_list.append(perc_pred_right)

                bord_x, bord_y, bord_z = borders_to_flip[:,0], borders_to_flip[:,1], borders_to_flip[:,2]

                surround_count, same_cts = get_number_similar_surroundings(cell_assign)

                old_cells = groupings[old_id].copy()
                old_cells[np.where(old_id == -1)] = n_celltype

                new_cells = groupings[new_id].copy()
                new_cells[np.where(new_id == -1)] = n_celltype
        
                percent_flipped.append(np.sum(np.equal(new_id,old_id))/number_of_border_pixels)

                toc = time()
                print('time to flip pixels ',toc-tic)
                n_iterations += 1
                num_iterations_most_inner += 1
            num_iterations_inner += 1
        num_iterations_outer += 1

    #remove border

    # evaluate metrics
    #perc_right, perc_pred_right = compute_metrics(cell_assign, t_map)
    #perc_right_cells, perc_pred_right_cells = compute_cellspec_metrics(cell_assign, t_map, groupings, t_cell)
    
    return cell_assign #, perc_right, perc_pred_right, perc_right_cells, perc_pred_right_cells, perc_correct_celltypes


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import json
import pickle as pkl
import pandas as pd


# In[67]:


def buildallentree(dist_from_top, name_or_code = True, col = False):
    with open('Allen_data/dend.json','r') as f:
        dict_ = json.load(f)

    def get_lists(dend, id_to_use,node_list, height_list):
        if 'children' in dend:
            for i in range(len(dend['children'])):
                #this is only because the color tag is not in 'node attributes'
                if id_to_use in dend['node_attributes'][0]:
                    node_list.append(dend['node_attributes'][0][id_to_use])
                else:
                    #node_list.append(dend['node_attributes'][0]['edgePar.col'])
                    #print(dend['node_attributes'][0]['edgePar.col'])
                    node_list.append('#FFFFFF')
                height_list.append(dend['node_attributes'][0]['height'])
                get_lists(dend['children'][i], id_to_use, node_list, height_list)

        else:
            #print(dend['leaf_attributes'][0][id_to_use])
            node_list.append(dend['leaf_attributes'][0][id_to_use])
            height_list.append(0)
            return dend['leaf_attributes'][0][id_to_use]
        return node_list, height_list
    node_list, height_list = get_lists(dict_, 'cell_set_accession', [], [])
    node_names, _ = get_lists(dict_, 'node_id', [], [])
    col_list, _ = get_lists(dict_, 'nodePar.col',[],[])
    #node_names, _ = get_lists(dict_, 'cell_set_designation', [])

    def process_nodes(node_list, height_list):
        seen_before = []
        celltype_lists = []
        height_lists = []
        celltype_list = []
        h_list = []
        for node, h in zip(node_list, height_list):
            if node not in seen_before:
                celltype_list.append(node)
                h_list.append(h)
                seen_before.append(node)
            else:
                celltype_lists.append(celltype_list)
                height_lists.append(h_list)
                celltype_list = np.array(celltype_list)
                new_loc = np.where(celltype_list == node)[0][0]
                celltype_list = celltype_list[0:new_loc+1]
                h_list = h_list[0:new_loc+1]
                celltype_list = list(celltype_list)
        celltype_lists.append(celltype_list)
        height_lists.append(h_list)
        for i in range(len(celltype_lists)):
            celltype_lists[i] = celltype_lists[i][::-1]
            height_lists[i] = height_lists[i][::-1]
        return celltype_lists, height_lists
    
    celltype_lists, height_lists = process_nodes(node_list, height_list)
    celltype_names, _ = process_nodes(node_names, height_list)
    celltype_cols, _ = process_nodes(col_list, height_list)
    #celltype_names = celltype_names[1:]

    allen_data = pd.read_csv('Allen_data/sample_cluster_probabilities.csv',
                            index_col=0)

    leaf_celltypes = allen_data.columns.to_numpy()

    classified_leaf = leaf_celltypes[np.argmax(allen_data.to_numpy(),axis=1)]

    def shorten_list(celltype_lists, height_lists, dist_from_top):
        shortened_lists = []
        for l, h in zip(celltype_lists, height_lists):
            loc_stop = np.where(np.array(h) > dist_from_top)[0][0]
            sub_list = l[0:loc_stop]
            if len(sub_list) == 0:
                sub_list = [l[0]]
            shortened_lists.append(sub_list)
        return shortened_lists
    shortened_lists = shorten_list(celltype_lists, height_lists, dist_from_top)
    shortened_names = shorten_list(celltype_names, height_lists, dist_from_top)
    shortened_cols = shorten_list(celltype_cols, height_lists, dist_from_top)

    for l, name, c in zip(shortened_lists,shortened_names, shortened_cols):
        if col == True:
            classified_leaf[classified_leaf == l[0]] = c[-1]
        if name_or_code:
            classified_leaf[classified_leaf == l[0]] = name[-1]
        else:
            classified_leaf[classified_leaf == l[0]] = l[-1]
    return classified_leaf


# In[68]:


def get_celltype_name_map(cells, names):
    cel_map = {}
    for i in range(len(cells)):
        if names[i] not in cel_map:
            cel_map[names[i]] = cells[i]
    return cel_map


# In[4]:




# In[5]:



# In[18]:




# In[19]:




# In[12]:



# In[15]:




# In[16]:




# In[ ]:





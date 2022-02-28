# JSTA: joint cell segmentation and cell type annotation for spatial transcriptomics
<p align="center">
  <img src=/images/JSTAOverview.png>
</p>
Initially, watershed based segmentation is performed and a cell level type classifier, parameterized by a deep neural network (DNN), is trained based on the NCTT data. The cell level classifier then assigns cell (sub)types (red and blue in this cartoon example). Based on the current assignment of pixels to cell (sub)types, a new DNN is trained to estimate the probabilities that each pixel comes from each of the possible (sub)types given the local RNA density at each pixel. In this example, two pixels that were initially assigned to the “red” cells got higher probability to be of a blue type. Since the neighbor cell is of type “blue” they were reassigned to that cell during segmentation update. Using the updated segmentation and the cell type classifier cell types are reassigned. The tasks of training, segmentation, and classification are repeated over many iterations until convergence. See the full manuscript here: https://doi.org/10.15252/msb.202010108

## Download and Install:  
### In terminal:
  ```git clone https://github.com/wollmanlab/JSTA.git```  
### Install python dependencies:  
  With pip:  
      ``` pip install -r CoreFunctions/requirements.txt ```  
  With conda:  
      ```conda create -n jsta -f CoreFunctions/environment.yml```  
      or  
      ```conda install --file CoreFunctions/requirements.txt```  
### Compile c files, and add current path to functions:  
  ```./install.sh```   
  
## Tutorials:
### tutorials/SimulatingData.ipynb
Simulate spatial transcriptomics data from a reference dataset:  
Files needed:  
  - scRNAseq Reference:
    - cells x genes matrix
  - Reference celltypes: 
    - cell type vector 
<p align="center">
  <img width="750", src=/images/SimulatedData.png>
</p>
Representative synthetic dataset of nuclei (black) and mRNAs, where each color represents a different gene (left). Ground truth boundaries of the cells. Each color represents a different cell (right). 

### tutorials/RunningJSTA.ipynb  
Run our quick implementation of density estimation, and segmentation with JSTA!  
Files needed:  
  - mRNA spots: 
    - spots x 4 matrix 
    - Columns: gene name, x, y, z  
    - Rows: Each mRNA spot
  - nuclei: 
    - pixels x 4 matrix; 
    - Columns: cell id, x, y, z 
    - Rows: Each pixel of nucleus 
  - scRNAseq Reference: 
    - cells x genes matrix
  - Reference celltypes: 
    - cell type vector 
<p align="center">
  <img src=/images/SegmentedHippocampus.png>
</p>  
High resolution cell type map of 133 cell (sub)types. Colors match those defined by Neocortical Cell Type Taxonomy. Scale bar is 500 microns.

### tutorials/FindSpatialDEGs.ipynb
Run our approach for finding spDEGs in your spatial data.

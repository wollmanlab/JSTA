# JSTA: joint cell segmentation and cell type annotation for spatial transcriptomics

![JSTA Overview](/images/JSTAOverview.png)
Initially, watershed based segmentation is performed and a cell level type classifier is trained based on the NCTT data. The cell level classifier then assigns cell (sub)types (red and blue in this cartoon example). Based on the current assignment of pixels to cell (sub)types, a new DNN is trained to estimate the probabilities that each pixel comes from each of the possible (sub)types given the local RNA density at each pixel. In this example, two pixels that were initially assigned to the “red” cells got higher probability to be of a blue type. Since the neighbor cell is of type “blue” they were reassigned to that cell during segmentation update. Using the updated segmentation and the cell type classifier cell types are reassigned. The tasks of training, segmentation, and classification are repeated over many iterations until convergence. See full manuscript here:  
## Download and Install:  
### In terminal:
  ```git clone https://github.com/wollmanlab/JSTA.git```  
### Install python dependencies:  
  With pip:  
      ``` pip install -r CoreFunctions/requirements.txt ```  
  With conda:  
      ```conda install --file CoreFunctions/requirements.txt```  
### Compile c files, and add current path to functions:  
  ```./install.sh```   
  
## Tutorials:
### tutorials/SimulatingData.ipynb


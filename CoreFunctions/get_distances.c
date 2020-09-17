//get_distances.c
#include <stdio.h>

void get_distances(const double **surroundings, const double **nuc_assign, const double **dists,const int num_pix, const int num_nuc, double **dist_mat){

	size_t i, j,k;
	double surr;
	double nuc_dist;

	//iterate through all pixels
	for(i=0;i<num_pix;i++){
		 //iterate through all surounding pixels
		for(j=0;j<27;j++){
			surr = surroundings[i][j];
			//iterate through nuclei
			for(k=0;k<num_nuc;k++){
				if(surr == nuc_assign[i][k]){
					dist_mat[i][j] = dists[i][k];
					break;
				}
			}
		}
	}
}


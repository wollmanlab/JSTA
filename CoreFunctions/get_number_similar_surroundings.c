//get_number_similar_surroundings.cpp
#include <stdio.h>

int get_raveled_index(int hei, int wid, int dep, int i_in, int j_in, int k_in){
	return i_in*wid*dep+j_in*dep+k_in;
}

void get_sur(const int *surroundings, int *surr_count, int *same_count, int height, int width, int depth){
        int i, j, k;
        int l, m, n;
        int current_pix;
        int surr_pix;
	int sur_ind, ind, real_ind;

	for(i=1;i<=height;i++){
		for(j=1;j<=width;j++){
			for(k=1;k<=depth;k++){
				ind = get_raveled_index(height+2, width+2, depth+2, i, j, k);
				real_ind = get_raveled_index(height, width, depth, i-1, j-1, k-1);
				current_pix = surroundings[ind];
				for(l=-1;l<2;l++){
					for(m=-1;m<2;m++){
						for(n=-1;n<2;n++){
							sur_ind = get_raveled_index(height+2, width+2, depth+2, i+l, j+m, k+n);
							surr_pix = surroundings[sur_ind];
							if (current_pix == surr_pix){
								same_count[real_ind]++;
							}else{
								if (surr_pix != -2)
									surr_count[real_ind]++;
							}
						}
					}
				}
			}
		}
	}	

}

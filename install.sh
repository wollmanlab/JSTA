cd CoreFunctions

#compile c functions
gcc -g -fPIC -shared -o get_distances.so get_distances.c
gcc -g -fPIC -shared -o get_number_similar_surroundings.so get_number_similar_surroundings.c

#add c functions to python path
sed -i 's?#REPLACE-WITH-PATH?'\"`pwd`\"'?' JSTA.py

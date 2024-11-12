I couldn't find any decently simple MPI implementation of Boids in C++.

Compile with Intel openmpi / gnu mvapich.

`mpicxx -np <n_tasks> ./boids_flock <n_boids> <n_iterations>`

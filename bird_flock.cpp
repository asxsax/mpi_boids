#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define MIN_POSITION 0.0f	// <<Boundary
#define MAX_POSITION 1000.0f	// Boundary>>
#define MIN_VELOCITY -50.0f
#define MAX_VELOCITY 50.0f

typedef struct {
    float x_pos, y_pos, z_pos;	// float of (x,y,z) of position
    float x_vel, y_vel, z_vel;  // float of (x,y,z) of velocity
} Bird;

void initialize_boids(Bird *birds, int num_birds, int rank);
void algorithm(Bird *flock, Bird *all_boids, int flock_count, int total_birds);
void position(Bird *flock, int flock_count);
void print_boids(Bird *birds, int num_birds, int start_index, const char *message);

int main(int argc, char *argv[]) {
    int num_birds, num_iterations;
    int rank, size;
    Bird *flock;
    Bird *all_boids = NULL;
    int flock_count, remainder, offset;
    int i;

    // MPI Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <number_of_birds> <number_of_iterations>\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    num_birds = atoi(argv[1]);
    num_iterations = atoi(argv[2]);

    flock_count = num_birds / size;	// Case 1: Even division of birds to tasks
    remainder = num_birds % size;	// Case 2: Odd division of birds to tasks
    if (rank < remainder) {
        flock_count++;
        offset = rank * flock_count;
    } else {
        offset = rank * flock_count + remainder;
    }

    // Allocate memory for birds
    flock = (Bird *)malloc(flock_count * sizeof(Bird));
    initialize_boids(flock, flock_count, rank);

    // Make sure all birds are initialized
    if (rank == 0) {
        all_boids = (Bird *)malloc(num_birds * sizeof(Bird));
    }

    // Number of items to send in comm for uneven boid number
    // Displacement of each ranks data in buffer
    // ref: https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node71.html
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int total_offset = 0;
    for (int r = 0; r < size; r++) {
        int lc = num_birds / size;
        int rem = num_birds % size;
        if (r < rem) {
            lc++;
            displs[r] = r * lc;
        } else {
            displs[r] = r * lc + rem;
        }
        recvcounts[r] = lc;
    }

    for (int r = 0; r < size; r++) {
        displs[r] *= sizeof(Bird);
        recvcounts[r] *= sizeof(Bird);
    }

    // Prelim-output; number of boids, iterations and tasks
    if (rank == 0) {
        printf("Number of birds: %d\n", num_birds);
        printf("Number of iterations: %d\n", num_iterations);
        printf("Number of MPI tasks: %d\n\n", size);
    }

    MPI_Gatherv(flock, flock_count * sizeof(Bird), MPI_BYTE,
                all_boids, recvcounts, displs, MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Initial data of all boids
    MPI_Barrier(MPI_COMM_WORLD);	// Barrier to wait for prelim-output
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            printf("Rank %d:\n", rank);
            print_boids(flock, flock_count, offset, "Initial positions and velocity");
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (i = 0; i < num_iterations; i++) {
        // Initial gather
	// Gatherv >> gather
        Bird *global_birds = (Bird *)malloc(num_birds * sizeof(Bird));
        MPI_Allgatherv(flock, flock_count * sizeof(Bird), MPI_BYTE,
                       global_birds, recvcounts, displs, MPI_BYTE,
                       MPI_COMM_WORLD);

	// Boid velocity and position update
        algorithm(flock, global_birds, flock_count, num_birds);
        position(flock, flock_count);

        free(global_birds);
    }

    // Final gather
    MPI_Gatherv(flock, flock_count * sizeof(Bird), MPI_BYTE,
                all_boids, recvcounts, displs, MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Final position & velocity for all boids
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            printf("Rank %d:\n", rank);
            print_boids(flock, flock_count, offset, "Final positions and velocities:");
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Clean up
    if (rank == 0) {
        free(all_boids);
    }
    free(recvcounts);		// Make sure this is free'd
    free(displs);		// Make sure this is free'd
    free(flock);
    MPI_Finalize();
    return 0;
}

// Initialize boids position and velocity.
// Explicit float values for velocity
// Spawn off screen (-1.0f, 1001.0f)
void initialize_boids(Bird *birds, int num_birds, int rank) {
    int i;
    srand(time(NULL) + rank);	// Random seed based off time
    for (i = 0; i < num_birds; i++) {
        //birds[i].x_pos = ((float)rand() / RAND_MAX) * (MAX_POSITION - MIN_POSITION) + MIN_POSITION;
	birds[i].x_pos = (rand() % 2 == 0) ? -1.0f : 1001.0f;	// Spawn boids offscreen
        birds[i].y_pos = ((float)rand() / RAND_MAX) * (MAX_POSITION - MIN_POSITION) + MIN_POSITION;
        birds[i].z_pos = ((float)rand() / RAND_MAX) * (MAX_POSITION - MIN_POSITION) + MIN_POSITION;
        birds[i].x_vel = 0.0f;
        birds[i].y_vel = 0.0f;
        birds[i].z_vel = 0.0f;
    }
}

// Main crux of boids algorithm; implements cohesion, alignment and separation
// Calculate velocity of boids given the three rules stated
void algorithm(Bird *flock, Bird *all_boids, int flock_count, int total_birds) {
    int i, j;
    for (i = 0; i < flock_count; i++) {
        Bird *b = &flock[i];
        float vx1 = 0.0f, vy1 = 0.0f, vz1 = 0.0f;
        float vx2 = 0.0f, vy2 = 0.0f, vz2 = 0.0f;
        float vx3 = 0.0f, vy3 = 0.0f, vz3 = 0.0f;

        // Rule 1: Boids fly towards the centre of mass of neighbouring boids
	// Make sure to exclude our own center of mass
        float pcx = 0.0f, pcy = 0.0f, pcz = 0.0f;
        for (j = 0; j < total_birds; j++) {
            if (&all_boids[j] != b) {
                pcx += all_boids[j].x_pos;
                pcy += all_boids[j].y_pos;
                pcz += all_boids[j].z_pos;
            }
        }
        pcx /= (total_birds - 1);
        pcy /= (total_birds - 1);
        pcz /= (total_birds - 1);
        vx1 = (pcx - b->x_pos) / 100;	// 1% as explained in the pseudocode
        vy1 = (pcy - b->y_pos) / 100;
        vz1 = (pcz - b->z_pos) / 100;

        // Rule 2: Boids fly together as a flock
        for (j = 0; j < total_birds; j++) {
            if (&all_boids[j] != b) {
                float distance = sqrtf(
                    powf(all_boids[j].x_pos - b->x_pos, 2) +
                    powf(all_boids[j].y_pos - b->y_pos, 2) +
                    powf(all_boids[j].z_pos - b->z_pos, 2));
                if (distance < 100.0f) {
                    vx2 -= (all_boids[j].x_pos - b->x_pos);
                    vy2 -= (all_boids[j].y_pos - b->y_pos);
                    vz2 -= (all_boids[j].z_pos - b->z_pos);
                }
            }
        }

        // Rule 3: Boids match velocity with boids in flock
        float pvx = 0.0f, pvy = 0.0f, pvz = 0.0f;
        for (j = 0; j < total_birds; j++) {
            if (&all_boids[j] != b) {
                pvx += all_boids[j].x_vel;
                pvy += all_boids[j].y_vel;
                pvz += all_boids[j].z_vel;
            }
        }
        pvx /= (total_birds - 1);
        pvy /= (total_birds - 1);
        pvz /= (total_birds - 1);
        vx3 = (pvx - b->x_vel) / 8;
        vy3 = (pvy - b->y_vel) / 8;
        vz3 = (pvz - b->z_vel) / 8;

        b->x_vel += vx1 + vx2 + vx3;
        b->y_vel += vy1 + vy2 + vy3;
        b->z_vel += vz1 + vz2 + vz3;

        // Check if velocity exceeds max/min
        if (b->x_vel > MAX_VELOCITY) b->x_vel = MAX_VELOCITY;
        if (b->x_vel < MIN_VELOCITY) b->x_vel = MIN_VELOCITY;
        if (b->y_vel > MAX_VELOCITY) b->y_vel = MAX_VELOCITY;
        if (b->y_vel < MIN_VELOCITY) b->y_vel = MIN_VELOCITY;
        if (b->z_vel > MAX_VELOCITY) b->z_vel = MAX_VELOCITY;
        if (b->z_vel < MIN_VELOCITY) b->z_vel = MIN_VELOCITY;
    }
}

// Update position of boids from n to n+1
void position(Bird *flock, int flock_count) {
    int i;
    for (i = 0; i < flock_count; i++) {
        flock[i].x_pos += flock[i].x_vel;
        flock[i].y_pos += flock[i].y_vel;
        flock[i].z_pos += flock[i].z_vel;

        // Check if position exceeds max/min
        if (flock[i].x_pos < MIN_POSITION) flock[i].x_pos = MIN_POSITION;
        if (flock[i].x_pos > MAX_POSITION) flock[i].x_pos = MAX_POSITION;
        if (flock[i].y_pos < MIN_POSITION) flock[i].y_pos = MIN_POSITION;
        if (flock[i].y_pos > MAX_POSITION) flock[i].y_pos = MAX_POSITION;
        if (flock[i].z_pos < MIN_POSITION) flock[i].z_pos = MIN_POSITION;
        if (flock[i].z_pos > MAX_POSITION) flock[i].z_pos = MAX_POSITION;
    }
}

// Print boid number, position and velocity to output using printf
void print_boids(Bird *birds, int num_birds, int start_index, const char *message) {
    int i;
    printf("%s\n", message);
    for (i = 0; i < num_birds; i++) {
        printf("Bird %d: Position=(%.2f, %.2f, %.2f) Velocity=(%.2f, %.2f, %.2f)\n",
               start_index + i, birds[i].x_pos, birds[i].y_pos, birds[i].z_pos,
               birds[i].x_vel, birds[i].y_vel, birds[i].z_vel);
    }
}


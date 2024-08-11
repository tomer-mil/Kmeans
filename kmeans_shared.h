#ifndef KMEANS_SHARED_H
#define KMEANS_SHARED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEFAULT_ITER 300
#define EPSILON 0.001
#define MAX_LINE_LENGTH 1000
#define MAX_DOUBLE 1.7976931348623157E+308

extern int dimension;

typedef struct {
    double* coordinates;
    int dimension;
    struct Cluster* cluster;
} Point;

typedef struct Cluster {
    Point centroid;
    Point sum_of_points;
    int number_of_points;
    int is_smaller_than_epsilon;
} Cluster;

// Function prototypes
void init_clusters(Cluster* clusters, Point* centroids, int k);
double distance_between_points(Point* p1, Point* p2);
void assign_point_to_cluster(Point* point, Cluster* cluster);
double update_centroid(Cluster* cluster);
Cluster* find_nearest_cluster(Point* point, Cluster* clusters, int k);
Cluster* run_kmeans(Point* centroids, Point* datapoints, int k, int n, int iter);
void free_memory(Point* datapoints, int n, Cluster* clusters, Point* centroids, int k);

#endif // KMEANS_SHARED_H
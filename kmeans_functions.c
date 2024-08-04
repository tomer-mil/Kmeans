#include "kmeans_shared.h"

void init_clusters(Cluster* clusters, Point* centroids, int k) {
    int i;

    for (i = 0; i < k; i++) {
        clusters[i].centroid = centroids[i];
        clusters[i].sum_of_points.coordinates = (double*) calloc(dimension, sizeof(double));
        if (!clusters[i].centroid.coordinates || clusters[i].sum_of_points.coordinates == NULL) {
            printf("An Error Has Occured\n");
            exit(1);
        }
        clusters[i].sum_of_points.dimension = dimension;
        clusters[i].number_of_points = 0;
        clusters[i].is_smaller_than_epsilon = 0;
    }
}

double distance_between_points(Point* p1, Point* p2) {
    double sum = 0.0;
    int i;
    for (i = 0; i < p1->dimension; i++) {
        double diff = p1->coordinates[i] - p2->coordinates[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void assign_point_to_cluster(Point* point, Cluster* cluster) {
    int i;
    if (point->cluster) {
        point->cluster->number_of_points--;
        for (i = 0; i < point->dimension; i++) {
            point->cluster->sum_of_points.coordinates[i] -= point->coordinates[i];
        }
    }

    point->cluster = cluster;
    cluster->number_of_points++;
    for (i = 0; i < point->dimension; i++) {
        cluster->sum_of_points.coordinates[i] += point->coordinates[i];
    }
}

double update_centroid(Cluster* cluster) {
    double max_diff = 0.0;
    int i;
    for (i = 0; i < cluster->centroid.dimension; i++) {
        double old_coord = cluster->centroid.coordinates[i];
        double diff;
        cluster->centroid.coordinates[i] = cluster->sum_of_points.coordinates[i] / cluster->number_of_points;
        diff = fabs(cluster->centroid.coordinates[i] - old_coord);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    cluster->is_smaller_than_epsilon = (max_diff < EPSILON);
    return max_diff;
}

Cluster* find_nearest_cluster(Point* point, Cluster* clusters, int k) {
    double min_distance = MAX_DOUBLE;
    Cluster* nearest_cluster = NULL;
    int i;

    for (i = 0; i < k; i++) {
        double distance = distance_between_points(point, &clusters[i].centroid);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_cluster = &clusters[i];
        }
    }

    return nearest_cluster;
}


void free_memory(Point* datapoints, int n, Cluster* clusters, Point* centroids, int k) {
    int i;
    for (i = 0; i < n; i++) {
        free(datapoints[i].coordinates);
    }
    free(datapoints);

    if (clusters) {
        for (i = 0; i < k; i++) {
            free(centroids[i].coordinates);
            free(clusters[i].centroid.coordinates);
            free(clusters[i].sum_of_points.coordinates);
        }
        free(clusters);
        free(centroids);
    }
}


Cluster* run_kmeans(Point* centroids, Point* datapoints, int k, int n, int max_iter) {
    
    printf("########## Entered kmeans_functions.c ##########");
    
    int iteration, done_clusters, i;
    Cluster* clusters;
    Cluster* nearest_cluster;

    clusters = (Cluster*) malloc(k * sizeof(Cluster));
    if (!clusters) {
        printf("An Error Has Occured\n");
        free_memory(datapoints, n, clusters, centroids, 0);
        return NULL; // TODO: check error handling (was 1)
    }
    
    printf("kmeans_functions: Entering init_clusters");

    init_clusters(clusters, centroids, k);

    printf("kmeans_functions: Finished init_clusters");

    iteration = 0;
    done_clusters = 0;

    while (iteration < max_iter && done_clusters < k) {
        done_clusters = 0;

        for (i = 0; i < n; i++) {
            nearest_cluster = find_nearest_cluster(&datapoints[i], clusters, k);
            assign_point_to_cluster(&datapoints[i], nearest_cluster);
        }

        for (i = 0; i < k; i++) {
            double centroid_diff = update_centroid(&clusters[i]);
            if (centroid_diff < EPSILON) {
                done_clusters++;
            }
        }

        iteration++;
    }
    return clusters;
}

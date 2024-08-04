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
    for (i = 0; i < dimension; i++) {
        double diff = p1->coordinates[i] - p2->coordinates[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void assign_point_to_cluster(Point* point, Cluster* cluster) {
    int i;

    printf("Entering assign_point_to_cluster\n");
    printf("Point dimension: %d, Cluster sum_of_points dimension: %d\n", point->dimension, cluster->sum_of_points.dimension);
    printf("Point address: %p, Cluster address: %p\n", (void*)point, (void*)cluster);

    if (!point || !cluster) {
        printf("Error: Null point or cluster\n");
        return;
    }

    printf("Point dimension: %d, Cluster sum_of_points dimension: %d\n", 
           point->dimension, cluster->sum_of_points.dimension);


    if (point->cluster) {

        printf("Point already assigned to cluster: %p\n", (void*)point->cluster);
        printf("Reducing number of points in old cluster from %d\n", point->cluster->number_of_points);

        point->cluster->number_of_points--;

        printf("Updating sum_of_points in old cluster\n");

        for (i = 0; i < dimension; i++) {
            printf("enterloop\n");
            printf("coordinate %d in cluster->sum_of_points: %f -= %f\n", i, point->cluster->sum_of_points.coordinates[i], point->coordinates[i]);

            point->cluster->sum_of_points.coordinates[i] -= point->coordinates[i];
        }
    }
    
    printf("LOOP\n");

    point->cluster = cluster;
    cluster->number_of_points++;
    for (i = 0; i < dimension; i++) {
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
        printf("finished distance_between_points\n");
        if (distance < min_distance) {
            min_distance = distance;
            nearest_cluster = &clusters[i];
        }
    }
    printf("Successfully found nearest cluster for point with header: %f", *point->coordinates);
    return nearest_cluster;
}


void free_memory(Point* datapoints, int n, Cluster* clusters, Point* centroids, int k) {
    int i;
    for (i = 0; i < n; i++) {
        free(datapoints[i].coordinates);
    }
    free(datapoints);

    printf("freed datapoints!\n");

    if (clusters) {
        for (i = 0; i < k; i++) {
            
            if (centroids && centroids[i].coordinates) free(centroids[i].coordinates);
            printf("freed centroids[i].coordinates\n");
            
            if (clusters[i].sum_of_points.coordinates) free(clusters[i].sum_of_points.coordinates);
            printf("freed clusters[i].sum_of_points.coordinates\n");
        }
        if (clusters) free(clusters);
        printf("freed clusters!\n");
        if (centroids) free(centroids);
        printf("freed centroids!\n");
    }
}


Cluster* run_kmeans(Point* centroids, Point* datapoints, int k, int n, int max_iter) {
    printf("Entering run_kmeans: k=%d, n=%d, max_iter=%d\n", k, n, max_iter);
    
    int iteration, done_clusters, i;
    Cluster* clusters;
    Cluster* nearest_cluster;

    clusters = (Cluster*) malloc(k * sizeof(Cluster));

    if (!clusters) {
        printf("An Error Has Occured\n");
        free_memory(datapoints, n, clusters, centroids, 0);
        return NULL; // TODO: check error handling (was 1)
    }

    init_clusters(clusters, centroids, k);

    iteration = 0;
    done_clusters = 0;

    printf("Entering while loop\n");
    while (iteration < max_iter && done_clusters < k) {
        done_clusters = 0;

        for (i = 0; i < n; i++) {
            nearest_cluster = find_nearest_cluster(&datapoints[i], clusters, k);
            printf("entering assign_point_to_cluster\n");
            
            assign_point_to_cluster(&datapoints[i], nearest_cluster);
            printf("finished assign_point_to_cluster\n");
        }

        printf("Finished nearest_cluster for loop\n");

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

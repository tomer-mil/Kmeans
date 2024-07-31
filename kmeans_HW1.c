#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define DEFAULT_ITER 200
#define EPSILON 0.001
#define MAX_LINE_LENGTH 1000
#define MAX_DOUBLE 1.7976931348623157E+308

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

/* Function prototypes */
Point* read_points_from_stdin(int* num_points, int* dimension);
void init_clusters(Cluster* clusters, Point* datapoints, int k, int dimension);
double distance_between_points(Point* p1, Point* p2);
void assign_point_to_cluster(Point* point, Cluster* cluster);
double update_centroid(Cluster* cluster);
Cluster* find_nearest_cluster(Point* point, Cluster* clusters, int k);
void free_memory(Point* datapoints, int num_points, Cluster* clusters, int k);
double parse_double(char** str);

int main(int argc, char* argv[]) {
    int k, max_iter, num_points, dimension;
    int iteration, done_clusters, i, j;
    Point* datapoints;
    Cluster* clusters;
    Cluster* nearest_cluster;

    if (argc < 2 || argc > 3) {
        printf("An Error Has Occured");
        return 1;
    }

    k = atoi(argv[1]);
    max_iter = (argc == 3) ? atoi(argv[2]) : DEFAULT_ITER;
    
    if (k <= 1 || k >= 1000) {
        printf("Invalid number of clusters!\n");
        return 1;
    }
    
    if (max_iter <= 1 || max_iter >= MAX_ITER) {
        printf("Invalid maximum iteration!\n");
        return 1;
    }

    num_points = 0;
    dimension = 0;
    datapoints = read_points_from_stdin(&num_points, &dimension);
    
    if (k >= num_points) {
        printf("Invalid number of clusters!\n");
        free_memory(datapoints, num_points, NULL, 0);
        return 1;
    }

    clusters = (Cluster*) malloc(k * sizeof(Cluster));
    if (!clusters) {
        printf("An Error Has Occured\n");
        free_memory(datapoints, num_points, NULL, 0);
        return 1;
    }

    init_clusters(clusters, datapoints, k, dimension);

    iteration = 0;
    done_clusters = 0;

    while (iteration < max_iter && done_clusters < k) {
        done_clusters = 0;

        for (i = 0; i < num_points; i++) {
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

    /* Print final centroids */
    for (i = 0; i < k; i++) {
        for (j = 0; j < dimension; j++) {
            printf("%.4f", clusters[i].centroid.coordinates[j]);
            if (j < dimension - 1) printf(",");
        }
        printf("\n");
    }

    free_memory(datapoints, num_points, clusters, k);
    return 0;
}

Point* read_points_from_stdin(int* num_points, int* dimension) {
    char line[MAX_LINE_LENGTH];
    Point* points = NULL;
    int dim;
    double* coords;
    char* line_ptr;

    *num_points = 0;
    *dimension = 0;

    while (fgets(line, MAX_LINE_LENGTH, stdin) != NULL) {
        if (line[0] != '\n') {  /* Ignore empty lines */
            (*num_points)++;
            points = (Point*) realloc(points, *num_points * sizeof(Point));
            if (!points) {
                printf("An Error Has Occurred\n");
                exit(1);
            }

            line_ptr = line;
            dim = 0;
            coords = NULL;

            while (*line_ptr != '\n' && *line_ptr != '\0') {
                dim++;
                coords = (double*) realloc(coords, dim * sizeof(double));
                if (!coords) {
                    printf("An Error Has Occurred\n");
                    exit(1);
                }
                coords[dim - 1] = parse_double(&line_ptr);
                if (*line_ptr == ',') line_ptr++; /* Skip comma */
            }

            if (*dimension == 0) {
                *dimension = dim;
            } else if (dim != *dimension) {
                printf("An Error Has Occurred\n");
                exit(1);
            }

            points[*num_points - 1].coordinates = coords;
            points[*num_points - 1].dimension = dim;
            points[*num_points - 1].cluster = NULL;
        }
    }

    return points;
}

double parse_double(char** str) {
    double result = 0.0;
    int sign = 1;
    int decimal_places = 0;
    int is_decimal = 0;

    /* Handle sign */
    if (**str == '-') {
        sign = -1;
        (*str)++;
    } 
    
    /* Parse integer part */
    while (**str >= '0' && **str <= '9') {
        result = result * 10.0 + (**str - '0');
        (*str)++;
    }

    /* Parse decimal part */
    if (**str == '.') {
        (*str)++;
        is_decimal = 1;
        while (**str >= '0' && **str <= '9') {
            result = result * 10.0 + (**str - '0');
            decimal_places++;
            (*str)++;
        }
    }

    /* Apply decimal point */
    if (is_decimal) {
        while (decimal_places > 0) {
            result /= 10.0;
            decimal_places--;
        }
    }

    return sign * result;
}

void init_clusters(Cluster* clusters, Point* datapoints, int k, int dimension) {
    int i, j;
    for (i = 0; i < k; i++) {
        clusters[i].centroid.coordinates = (double*) malloc(dimension * sizeof(double));
        clusters[i].sum_of_points.coordinates = (double*) malloc(dimension * sizeof(double));
        if (!clusters[i].centroid.coordinates || !clusters[i].sum_of_points.coordinates) {
            printf("An Error Has Occured\n");
            exit(1);
        }

        for (j = 0; j < dimension; j++) {
            clusters[i].centroid.coordinates[j] = datapoints[i].coordinates[j];
            clusters[i].sum_of_points.coordinates[j] = 0.0;
        }

        clusters[i].centroid.dimension = dimension;
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

void free_memory(Point* datapoints, int num_points, Cluster* clusters, int k) {
    int i;
    for (i = 0; i < num_points; i++) {
        free(datapoints[i].coordinates);
    }
    free(datapoints);

    if (clusters) {
        for (i = 0; i < k; i++) {
            free(clusters[i].centroid.coordinates);
            free(clusters[i].sum_of_points.coordinates);
        }
        free(clusters);
    }
}

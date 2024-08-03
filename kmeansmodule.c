# define PY_SSIZE_T_CLEAN
# include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans_functions.c"

// maybe delete definitions, keep them in kmeans_functions.c
#define iter 1000
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

// args = [point[],centroids[], iter]
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject* datapoints_lst;
    PyObject* centroids_lst;
    PyObject* python_centroids;
    PyObject* clusters;
    int iter, k, n;

    if (!PyArg_ParseTuple(args, "OOiii", &datapoints_lst, &centroids_lst, &iter, &k, &n)) {
        return NULL;
    }

    // parse initialized clusters' centroids from python to C
    Point* centroids = PyPointsLst_AsPointsArr(&centroids_lst, k);
    if (!centroids)  {
        return NULL;
    }

    // parse datapoints from python to C
    Point* datapoints = PyPointsLst_AsPointsArr(&datapoints_lst, n);
    if (!datapoints)  {
        return NULL;
    }
    
    // clusters is an array of Points representing the clusters
    clusters = run_kmeans(centroids, datapoints, k, iter);

    // building python list containing the clusters' centroids
    PyCentroids_FromClusters(&clusters, python_centroids, k);
    
    // free memory before exit
    free_memory(datapoints, n, clusters, centroids, k);
    return python_clusters;
}


static Point* PyPointsLst_AsPointsArr(PyObject *points_lst, int n) {
    PyObject *item;

    Point* points = (Point *)malloc(n * sizeof(Point));
    if (points == NULL) {
        printf("Memory allocation failed. Exiting.\n");
        return NULL;
    }

    // after finishing this loop the points array should be initialized properly as C struct Point array
    int i;
    int dimension = sizeof(points[0])
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_lst, i);
        PyPoint_AsPoint(item, points[i], dimension);
    }
    return points;
}

static void PyPoint_AsPoint(PyObject *item, Point *point, int dimension) {
    PyObject *coordinates_lst;   
    PyObject *coordinate_item;
    double coordinate;
    
    // parsing a Point into coordinates list, dimension and Cluster 
    if (!PyArg_ParseTuple(item, "O", &coordinates_lst)) {
        return NULL;
    }
    
    double* coords = (double*) malloc(dimension * sizeof(double));
    if (coords == NULL) {
        printf("Memory allocation failed. Exiting.\n");
        return NULL;
    }
    
    // parsing the point's coordinates list into a C double array
    int i;
    for (i = 0; i < dimension; i++) {
        coordinate_item = PyList_GetItem(coordinates_lst, i);
        coordinate = PyFloat_AsDouble(coordinate_item);
        coords[i] = coordinate;
    }

    point->coordinates = coords;
    point->dimension = dimension;
}

// parse centroids list in Python from clusters array in C
static void PyCentroids_FromClusters(Cluster* clusters, PyObject* python_centroids, int k) {
    PyObject* python_coordinate;
    double* cluster_coordinates;
    int dimension = sizeof(clusters[0]);

    python_centroids = PyList_New(k);
    for (int i = 0; i < k; ++i) /* parse outer list */
    {   
        PyObject* python_coordinates = PyList_New(dimension);
        cluster_coordinates = clusters[i].centroid;
        for (int j = 0; j < dimension; ++j) /* parse each centroids (inner lists) */
        {   
            python_coordinate = PyFloat_FromDouble(coordinates[j]);
            PyList_SetItem(python_coordinates, j, python_coordinate);
        }
        PyList_SetItem(python_centroids, i, python_coordinates);
    }
    // now the python_centroids should be updated with the centroids
}

static PyMethodDef kmeans_FunctionsTable[] = {
    {
        "run_kmeans", // name exposed to Python
        fit, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "runs the kmeans algoritm as requested, except from centroids initialization" // documentation
    }, {
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef kmeans_Module = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",     // name of module exposed to Python
    "kmeans Python wrapper for kmeans C implementation.", // module documentation
    -1,
    kmeans_FunctionsTable
};

PyMODINIT_FUNC PyInit_kmeans(void) {
    return PyModule_Create(&kmeans_Module);
}
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include "kmeans_shared.h"

int dimension;

static int PyPoint_AsPoint(PyObject *item, Point *point) {
    PyObject* coordinate_item;
    double coordinate;

    double* coords = (double*) malloc(dimension * sizeof(double));
    if (coords == NULL) {
        return 0;
    }
    
    // Parsing the point's coordinates list into a C double array
    Py_ssize_t i;
    for (i = 0; i < dimension; i++) {
        coordinate_item = PyList_GetItem(item, i);
        coordinate = PyFloat_AsDouble(coordinate_item);
        coords[i] = coordinate;
    }

    point->coordinates = coords;
    point->dimension = dimension;
    point->cluster = NULL;
    return 1;
}


static Point* PyPointsLst_AsPointsArr(PyObject *points_lst, int n) {

    PyObject *item;

    Point* points = (Point*) malloc(n * sizeof(Point));
    if (points == NULL) {
        return NULL;
    }

    // After finishing this loop the points array should be initialized properly as a C struct Point array
    Py_ssize_t i;
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_lst, i);
        int res = PyPoint_AsPoint(item, &(points[i])); 
        if (res == 0)
            return NULL; // TODO: handle ret error 
    }
    return points;
}


// Generate a centroids Python list from a C clusters array
static PyObject* PyCentroids_FromClusters(Cluster* clusters, int k) {
    PyObject* python_centroids;

    python_centroids = PyList_New(k);

    // Parse outer list
    for (int i = 0; i < k; ++i) {
        PyObject* python_coordinates = PyList_New(dimension);
        double* cluster_coordinates = clusters[i].centroid.coordinates;

        // Parse each centroids (inner lists)
        for (int j = 0; j < dimension; ++j) {
            PyObject* python_coordinate = PyFloat_FromDouble(cluster_coordinates[j]);
            PyList_SetItem(python_coordinates, j, python_coordinate);
        }
        PyList_SetItem(python_centroids, i, python_coordinates);
    }
    // python_centroids should be updated with the centroids
    return python_centroids;
}


static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject* datapoints_lst;
    PyObject* centroids_lst;
    PyObject* python_centroids = NULL;
    Cluster* clusters = NULL;
    Point* datapoints = NULL;
    Point* centroids = NULL;
    unsigned long iter;
    int k, n, d;

    // Parse input data from Python
    if (!PyArg_ParseTuple(args, "OOiiii", &datapoints_lst, &centroids_lst, &iter, &k, &n, &d)) {
        return NULL;
    }

    // Set dimension global variable
    dimension = d;

    // Parse Python initialized clusters' centroids to C
    centroids = PyPointsLst_AsPointsArr(centroids_lst, k);
    if (!centroids)  {
        return NULL;
    }

    // Parse Python datapoints list to C array
    datapoints = PyPointsLst_AsPointsArr(datapoints_lst, n);
    if (!datapoints) {
        return NULL;
    }

    // `clusters` is an array of Points representing the clusters
    clusters = run_kmeans(centroids, datapoints, k, n, iter);

    // Generate Python list with the clusters' centroids
    python_centroids = PyCentroids_FromClusters(clusters, k);
    
    // Free memory before exit
    free_memory(datapoints, n, clusters, centroids, k);
    return python_centroids;
}


static PyMethodDef kmeans_FunctionsTable [] = {
    {
        "python_fit", // Name exposed to Python
        (PyCFunction) fit, // C wrapper function
        METH_VARARGS, // Received variable args (but really just 1)
        "Runs the K-means algorithm with provided clusters as requested" // Documentation
    },
    {
        NULL, NULL, 0, NULL
    }
};


// Modules definition
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",     // Name of module exposed to Python
    "K-means Python wrapper for K-means C implementation.", // Module documentation
    -1,
    kmeans_FunctionsTable
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&kmeansmodule);
}
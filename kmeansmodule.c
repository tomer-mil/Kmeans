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
        printf("Memory allocation failed. Exiting.\n");
        return 0;
    }
    
    // parsing the point's coordinates list into a C double array
    Py_ssize_t i;
    for (i = 0; i < dimension; i++) {
        coordinate_item = PyList_GetItem(item, i);
        coordinate = PyFloat_AsDouble(coordinate_item);
        coords[i] = coordinate;
    }

    point->coordinates = coords;
    point->dimension = dimension;
    return 1;
}


static Point* PyPointsLst_AsPointsArr(PyObject *points_lst, int n) {

    PyObject *item;

    Point* points = (Point*) malloc(n * sizeof(Point));
    if (points == NULL) {
        printf("Memory allocation failed. Exiting.\n");
        return NULL;
    }

    // after finishing this loop the points array should be initialized properly as C struct Point array
    Py_ssize_t i;
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_lst, i);
        int res = PyPoint_AsPoint(item, &(points[i])); 
        if (res == 0)
            return NULL; // TODO: handle ret error 
    }
    return points;
}


// parse centroids list in Python from clusters array in C
static PyObject* PyCentroids_FromClusters(Cluster* clusters, int k) {
    PyObject* python_centroids;

    python_centroids = PyList_New(k);
    for (int i = 0; i < k; ++i) /* parse outer list */
    {   
        PyObject* python_coordinates = PyList_New(dimension);
        double* cluster_coordinates = clusters[i].centroid.coordinates;
        for (int j = 0; j < dimension; ++j) /* parse each centroids (inner lists) */
        {   
            PyObject* python_coordinate = PyFloat_FromDouble(cluster_coordinates[j]);
            PyList_SetItem(python_coordinates, j, python_coordinate);
        }
        PyList_SetItem(python_centroids, i, python_coordinates);
    }
    // now the python_centroids should be updated with the centroids
    return python_centroids;
}


// args = [point[],centroids[], iter]
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject* datapoints_lst;
    PyObject* centroids_lst;
    PyObject* python_centroids = NULL;
    Cluster* clusters = NULL;
    Point* datapoints = NULL;
    Point* centroids = NULL;
    unsigned long iter;
    int k, n, d;

    if (!PyArg_ParseTuple(args, "OOiiii", &datapoints_lst, &centroids_lst, &iter, &k, &n, &d)) {
        return NULL;
    }
    dimension = d;
    // parse initialized clusters' centroids from python to C
    centroids = PyPointsLst_AsPointsArr(centroids_lst, k);
    if (!centroids)  {
        return NULL;
    }

    // parse datapoints from python to C
    datapoints = PyPointsLst_AsPointsArr(datapoints_lst, n);
    if (!datapoints)  {
        return NULL;
    }

    // clusters is an array of Points representing the clusters
    clusters = run_kmeans(centroids, datapoints, k, n, iter);

    printf("Passed run_kmeans in fit function\n");

    // building python list containing the clusters' centroids
    python_centroids = PyCentroids_FromClusters(clusters, k);
    
    // free memory before exit
    free_memory(datapoints, n, clusters, centroids, k);
    return python_centroids;
}




static PyMethodDef kmeans_FunctionsTable [] = {
    {
        "python_fit", // name exposed to Python
        (PyCFunction) fit, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "runs the kmeans algoritm as requested, except from centroids initialization" // documentation
    }, {
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",     // name of module exposed to Python
    "kmeans Python wrapper for kmeans C implementation.", // module documentation
    -1,
    kmeans_FunctionsTable
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&kmeansmodule);
}
# define PY_SSIZE_T_CLEAN
# include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans_functions.c"

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

// args = [point[],centroids[], max_iter]
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject* datapoints_lst;
    PyObject* centroids_lst;
    PyObject* python_centroids;
    int max_iter;
    // TODO: get max_iter and other attr
    if (!PyArg_ParseTuple(args, "OOi", &datapoints_lst, &centroids_lst, &max_iter)) {
        return NULL;
    }

    int k = PyObject_Length(centroids_lst);
    if (k < 0) {
        return NULL;
    }

    Point* centroids = PyPointsLst_AsPointsArr(&centroids_lst);
    if (!centroids)  {
        return NULL;
    }

    Point* datapoints = PyPointsLst_AsPointsArr(&datapoints_lst);
    if (!datapoints)  {
        return NULL;
    }
    
    centroids = run_kmeans(centroids, datapoints, k, max_iter);

    python_centroids = PyList_New(k);
    for (int i = 0; i < k; ++i)
    {
        python_obj = Py_BuildValue("O", centroids[i]);
        PyList_SetItem(python_centroids, i, python_obj);
    }

    free_memory(datapoints, num_points, clusters, centroids, k);
    return python_centroids;
}


static Point* PyPointsLst_AsPointsArr(PyObject *points_lst) {
    PyObject *item;
    
    int n = PyObject_Length(points_lst);
    if (n < 0) {
        return NULL;
    }

    Point* points = (Point *)malloc(n * sizeof(Point));
    if (points == NULL) {
        printf("Memory allocation failed. Exiting.\n");
        return NULL;
    }

    // after finishing this loop the points array should be initiated properly
    int i;
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_lst, i);
        PyPoint_AsPoint(item, points[i]);
    }
    return points;
}

static void PyPoint_AsPoint(PyObject *item, Point *point) {
    PyObject *coordinates_lst;   
    PyObject *cluster;
    PyObject *coordinate_item;
    int dimension;
    double coordinate;
    
    // parsing a Point into coordinates list, dimension and Cluster 
    if (!PyArg_ParseTuple(item, "OiO", &coordinates_lst, &dimension, &cluster)) {
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
    // TODO: point.cluster
}

static PyObject* GetList(PyObject* self, PyObject* args)
{
    int N,r;
    PyObject* python_val;
    PyObject* python_int;
    if (!PyArg_ParseTuple(args, "i", &N)) {
        return NULL;
    }
    if (N < 3) {
        PyErr_SetString(PyExc_ValueError, "List length must be greater than 3");
        return NULL;
    }

    python_val = PyList_New(N);
    for (int i = 0; i < N; ++i)
    {
        r = i;
        python_int = Py_BuildValue("i", r);
        PyList_SetItem(python_val, i, python_int);
    }
    return python_val;
}
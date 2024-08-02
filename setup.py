from setuptools import Extension, setup

module = Extension("mykmeanssp",
                  sources=[
                    'kmeans_functions.c',
                    'kmeansmodule.c'
                  ])
setup(name='mykmeanssp',
     version='1.0',
     description='Python wrapper for kmeans C extension',
     ext_modules=[module])
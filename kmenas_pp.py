import numpy as np
import pandas as pd
import sys
from math import dist
from enum import Enum

MAX_ITER = 1000
DEFAULT_ITER = 200
EPSILON = 0.001

INVALID_K_ERROR_MSG = "Invalid number of clusters!"
INVALID_ITER_ERROR_MSG = "Invalid maximum iteration!"
INVALID_FILE_NAME_ERROR_MSG = "Invalid file names!"
INVALID_EPSILON_ERROR_MSG = "Invalid epsilon!"
GENERAL_ERROR_MSG = "An Error Has Occurred"

# Example of a script call: `python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt`
class ArgumentIndex(Enum):
	PYTHON_CALL = 0
	PY_FILE = 1
	K = 2
	ITER = 3
	EPSILON = 4
	FILE1_URL = 5
	FILE2_URL = 6

class Point:
	coordinates: np.ndarray
	cluster: None

	def __init__(self, coordinates: np.ndarray[np.float64]):
		self.coordinates = coordinates.copy()
		self.cluster = None

	def __add__(self, other):
		return self.coordinates + other.coordinates

	def __iadd__(self, other):
		self.coordinates = self.coordinates + other.coordinates
		return self

	def __isub__(self, other):
		self.coordinates = self.coordinates - other.coordinates
		return self

	def __repr__(self):
		return np.array2string(a=self.coordinates, separator=',', precision=4, sign='-')

	def __str__(self):
		return np.array2string(a=self.coordinates, separator=',', precision=4, sign='-')

	def __copy__(self):
		return Point(coordinates=self.coordinates)

	# Assigns/Unassigns a point to/from a cluster
	def assign_to_cluster(self, cluster) -> None:
		if self.cluster:
			self.cluster.remove_point(point=self)

		cluster.add_point(point=self)
		self.cluster = cluster

	# Distance between 2 points
	def distance_to_point(self, other_point) -> float:
		return dist(self.get_coordinates(), other_point.get_coordinates())

	# Divide a point by a number
	def divide(self, divisor: float):
		self.coordinates = [x/divisor for x in self.coordinates]

	# Returns a copy of the coordinates list from a point
	def get_coordinates(self, is_copy: bool = False) -> list:
		return self.coordinates.copy() if is_copy else self.coordinates

	# Sets a point's coordinates from a given list of floats
	def set_coordinates(self, new_coordinates: list):
		self.coordinates = new_coordinates


class Cluster:

	centroid: Point
	sum_of_points: Point
	number_of_points = 0
	is_smaller_than_epsilon = False

	def __init__(self, init_value: Point):
		self.centroid = Point(coordinates=init_value.get_coordinates(is_copy=True))  # Generates a centroid from a given point

		# Initializes sum_of_points to an empty Point
		dimension = len(self.centroid.get_coordinates())
		self.sum_of_points = Point(coordinates=([0.0] * dimension))

	def __repr__(self):
		return self.centroid.__repr__()

	def __str__(self):
		return self.centroid.__str__()

	# Update a cluster's centroid given the cluster's sum and number of points (calcs the mean of this cluster)
	# Returns a float representing the difference between the previous centroid and the new one.
	def update_centroid(self) -> float:
		old_centroid = Point(coordinates=self.centroid.get_coordinates())
		self.centroid.set_coordinates(new_coordinates=self.sum_of_points.get_coordinates(is_copy=True))
		self.centroid.divide(divisor=self.number_of_points)

		centroid_diff = self.centroid.distance_to_point(other_point=old_centroid)

		return centroid_diff

	# Removes an existing point from a cluster
	def remove_point(self, point: Point):
		self.sum_of_points -= point
		self.number_of_points -= 1

	# Adds a new point to a cluster
	def add_point(self, point: Point):
		self.sum_of_points += point
		self.number_of_points += 1

	# Calculates the distance between the cluster's centroid (which defines it) and a given point
	def calc_distance_from_point(self, point: Point) -> float:
		return point.distance_to_point(other_point=self.centroid)

class Kmeans:

	clusters: list
	datapoints: list
	max_iter: int
	k: int

	def __repr__(self):
		kmeans_description =\
			f"K: {self.k}, " \
			f"No. of Datapoints: {len(self.datapoints)}, " \
			f"Iter: {self.max_iter}, " \
			f"No. of Clusters: {len(self.clusters)}, " \
			f"Clusters: {self.clusters}"
		return kmeans_description

	def __str__(self):

		kmeans_description = f"""
			K: {self.k}, 
			No. of Datapoints: {len(self.datapoints)}, 
			Iter: {self.max_iter}, 
			No. of Clusters: {len(self.clusters)}, 
			Clusters: {self.clusters}, 
		"""

		return kmeans_description

	# Reader for the data.txt file
	@staticmethod
	def get_points_from_df(input_df: pd.DataFrame) -> list:
		points = []
		# TODO: LAST STOP (23.07 [Tuesday]). Next up: Convert df to np array or something similar to create Point classes.
		for index, point in input_df.iterrows():
			pass

		with open(input_df, 'r') as data:
			for line in data:
				if line:  # Avoid blank lines
					point_coordinates = [float(x) for x in line.split(',')]
					points.append(Point(coordinates=point_coordinates))
		return points

	def __init__(self, k: int, data_file_url: str, max_iter: int):

		self.datapoints = self.get_points_from_df(input_df=data_file_url)

		# assert self.is_natural(k) and 1 < k < len(self.datapoints), "Invalid number of clusters!"
		# assert self.is_natural(max_iter) and 1 < max_iter < MAX_ITER, "Invalid maximum iteration!"

		self.max_iter = max_iter
		self.k = k
		self.clusters = list()

	# Initializes the k clusters from the first k datapoints
	def init_clusters(self):
		for i in range(self.k):
			init_point = Point(coordinates=self.datapoints[i].get_coordinates())
			self.clusters.append(Cluster(init_value=init_point))

	# The output for the assignment
	def print_clusters(self):
		for cluster in self.clusters:
			print(cluster)

	# Bulk calculates the distances from a given point to all k clusters.
	# Returns a list of tuples, each with the following format:
	# 	(distance_from_cluster_to_point, cluster)
	def calc_distances_from_point_to_all_clusters(self, point: Point) -> list:
		return [(cluster.calc_distance_from_point(point=point), cluster) for cluster in self.clusters]

	# Returns the nearest cluster from a given point
	def find_nearest_cluster_to_point(self, point: Point) -> Cluster:

		distance_cluster_tuples_list = self.calc_distances_from_point_to_all_clusters(point=point)

		# Returns the second element of the tuple (a cluster)
		# which its distance to the given point is the minimum ('key' is set to address only the distance parameter)
		return min(distance_cluster_tuples_list, key=lambda t: t[0])[1]

	def run_kmeans(self):

		# Init k clusters
		self.init_clusters()

		# Flags for termination conditions
		iteration_number = 0
		done_clusters = 0

		while (iteration_number < self.max_iter) and (done_clusters < self.k):
			done_clusters = 0

			# For each datapoint
			for point in self.datapoints:

				# Find its nearest cluster
				nearest_cluster = self.find_nearest_cluster_to_point(point=point)

				# Assign the datapoint to that cluster (unassign from another cluster if needed)
				point.assign_to_cluster(cluster=nearest_cluster)

			# Update clusters' centroid value per this iteration
			for cluster in self.clusters:

				# Calculate the size of change done on this iteration
				centroid_diff = cluster.update_centroid()

				# If the change is small enough, flag (count) it
				if centroid_diff < EPSILON:
					done_clusters += 1
			iteration_number += 1
		self.print_clusters()

	class CommandLineReader:

		k: str
		datapoints_df: pd.DataFrame
		n: int
		iter: int
		epsilon: float

		@classmethod
		# Checks whether the string `n` represents a natural number(0 excluded)
		def is_natural(n) -> bool:
			return str(n).isdigit() and float(n) == int(n) and int(n) > 0
		
		@classmethod
		# Retrieves the file URL suffix (hence type)
		def get_file_suffix(file_url: str) -> str:
			return file_url.split(".")[-1]
		
		@classmethod
		# Validates that the passed URL is either a txt or csv file.
		def is_valid_file_url(url: str) -> bool:
			return Kmeans.CommandLineReader.get_file_suffix(file_url=url) in {".csv", ".txt"}

		@classmethod
		# Validates to correctness of the passed cmd command and returns errors as requested in the assignment
		def validate_cmd_arguments(arguments_list: list) -> None:
			# Assert number of arguments passed
			assert len(arguments_list) == 6, GENERAL_ERROR_MSG
			# Assert K
			assert Kmeans.CommandLineReader.is_natural(arguments_list[ArgumentIndex.K.value]), INVALID_K_ERROR_MSG
			# Assert iter
			assert (Kmeans.CommandLineReader.is_natural(arguments_list[ArgumentIndex.ITER.value]) and 1 < arguments_list[ArgumentIndex.ITER.value] < MAX_ITER), INVALID_ITER_ERROR_MSG
			# Assert epsilon
			assert (arguments_list[ArgumentIndex.EPSILON.value].isnumeric() and arguments_list[ArgumentIndex.EPSILON.value] >= 0), INVALID_EPSILON_ERROR_MSG
			# Assert file names
			assert (Kmeans.CommandLineReader.is_valid_file_url(url=arguments_list[ArgumentIndex.FILE1_URL.value]) and Kmeans.CommandLineReader.is_valid_file_url(url=arguments_list[ArgumentIndex.FILE2_URL.value])), INVALID_FILE_NAME_ERROR_MSG

		@classmethod
		# Returns an inner-joined and sorted DataFrame from the passed files' URLs
		def generate_df(file1_url: str, file2_url: str) -> pd.DataFrame:
			file1_df = pd.read_csv(filepath_or_buffer=file1_url, header=None)
			file2_df = pd.read_csv(filepath_or_buffer=file2_url, header=None)

			datapoints_df = file1_df.join(file2_df, how="inner", sort=True)

			return datapoints_df
		

		def __init__(self, cmd_input: list):

			if len(cmd_input) == 5:
				# Assuming no `iter` provided, adding default iteration value
				cmd_input = cmd_input[ArgumentIndex.PYTHON_CALL.value:ArgumentIndex.K.value] + [DEFAULT_ITER] + cmd_input[ArgumentIndex.K.value:]
			
			Kmeans.CommandLineReader.validate_cmd_arguments(arguments_list=cmd_input)

			self.k = int(cmd_input[ArgumentIndex.K.value])
			self.iter = int(cmd_input[ArgumentIndex.ITER.value])
			self.epsilon = float(cmd_input[ArgumentIndex.EPSILON.value])
			self.datapoints_df = Kmeans.CommandLineReader.generate_df(file1_url=cmd_input[ArgumentIndex.FILE1_URL.value], file2_url=cmd_input[ArgumentIndex.FILE2_URL.value])
			self.n = self.datapoints_df.size()


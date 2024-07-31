import sys
from math import dist

MAX_ITER = 1000
DEFAULT_ITER = 200
EPSILON = 0.001


class Point:
	coordinates: list
	cluster: None

	def __init__(self, coordinates: list[float]):
		self.coordinates = coordinates.copy()
		self.cluster = None

	def __add__(self, other):
		return [sum(x) for x in zip(self.coordinates, other.coordinates)]

	def __iadd__(self, other):
		self.coordinates = [sum(x) for x in zip(self.coordinates, other.coordinates)]
		return self

	def __isub__(self, other):
		self.coordinates = [x[0] - x[1] for x in zip(self.coordinates, other.coordinates)]
		return self

	def __repr__(self):
		return ",".join(f"{x:0.4f}" for x in self.coordinates)

	def __str__(self):
		return ",".join(f"{x:0.4f}" for x in self.coordinates)

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

	@staticmethod
	def __is_natural(n) -> bool:
		return str(n).isdigit() and float(n) == int(n) and int(n) > 0

	# Reader for the data.txt file
	@staticmethod
	def get_points_from_data(input_file_path: str) -> list:
		points = []
		with open(input_file_path, 'r') as data:
			for line in data:
				if line:  # Avoid blank lines
					point_coordinates = [float(x) for x in line.split(',')]
					points.append(Point(coordinates=point_coordinates))
		return points

	def __init__(self, k: int, data_file_url: str, max_iter: int):

		self.datapoints = self.get_points_from_data(input_file_path=data_file_url)

		assert self.__is_natural(k) and 1 < k < len(self.datapoints), "Invalid number of clusters!"
		assert self.__is_natural(max_iter) and 1 < max_iter < MAX_ITER, "Invalid maximum iteration!"

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


if __name__ == '__main__':
	k = int(sys.argv[1])
	max_iter = int(sys.argv[2]) if len(sys.argv) == 4 else DEFAULT_ITER
	data_url = sys.argv[-1]

	kmeans_runner = Kmeans(k=k, data_file_url=data_url, max_iter=max_iter)
	kmeans_runner.run_kmeans()

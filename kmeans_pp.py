import sys
from enum import Enum, IntEnum
import pandas as pd
import numpy as np
import mykmeanssp

MAX_ITER = 1000
DEFAULT_ITER = 200
EPSILON = 0.001

NP_RANDOM_SEED = 1234

IS_CENTROID_COL_NAME = 'isCentroid'
D_VALUE_COL_NAME = 'D'


class ErrorMessages(Enum):
	INVALID_K_ERROR_MSG = "Invalid number of clusters!"
	INVALID_ITER_ERROR_MSG = "Invalid maximum iteration!"
	INVALID_FILE_NAME_ERROR_MSG = "Invalid file names!"
	INVALID_EPSILON_ERROR_MSG = "Invalid epsilon!"
	GENERAL_ERROR_MSG = "An Error Has Occurred"


class Argument(IntEnum):
	PY_FILE = 0
	K = 1
	ITER = 2
	EPSILON = 3
	FILE1_URL = 4
	FILE2_URL = 5
	OTHER = -1


class CommandLineReader:
	"""
	The CommandLineReader class is used to read the command line arguments, validate their values according to the assignment's
	guidelines and handling potential errors.
	"""
	k: int
	iter: int
	epsilon: float
	file1_url: str
	file2_url: str

	def __init__(self, cmd_input):
		if len(cmd_input) == 5:
			# Assuming no `iter` provided, adding default iteration value
			cmd_input = cmd_input[Argument.PY_FILE.value:Argument.ITER.value] + [DEFAULT_ITER] + cmd_input[Argument.ITER.value:]

		self.validate_cmd_arguments(arguments_list=cmd_input)

		self.k = int(cmd_input[Argument.K.value])
		self.iter = int(cmd_input[Argument.ITER.value])
		self.epsilon = float(cmd_input[Argument.EPSILON.value])
		self.file1_url = str(cmd_input[Argument.FILE1_URL.value])
		self.file2_url = str(cmd_input[Argument.FILE2_URL.value])

	@staticmethod
	# Checks whether the string `n` represents a natural number (0 excluded)
	def is_natural(n):
		return str(n).isdigit() and float(n) == int(n) and int(n) > 0

	@staticmethod
	# Checks whether the string `n` represents a float number
	def is_valid_float(n):
		try:
			float(n)
			return True
		except ValueError:
			return False
		#
		# tmp = n.replace('.', '', 1)
		# return tmp.isdigit() and float(tmp) == int(tmp)

	@classmethod
	# Checks if the input epsilon argument is a non-negative number
	def is_valid_epsilon(cls, epsilon):
		if cls.is_valid_float(epsilon):
			return float(epsilon) >= 0
		return False

	@classmethod
	# Checks if the input K is a natural number greater than 1
	def is_valid_k(cls, k):
		if cls.is_natural(k):
			return int(k) > 1
		return False

	@classmethod
	# Checks if the input iter is a natural number smaller than the MAX_ITER defined
	def is_valid_iter(cls, input_iter):
		if cls.is_natural(input_iter):
			return 1 < int(input_iter) < MAX_ITER
		return False

	@classmethod
	# Retrieves the file URL suffix (hence type)
	def get_file_suffix(cls, file_url):
		return file_url[-4:]

	@classmethod
	# Validates that the passed URL is either a txt or csv file.
	def is_valid_file_url(cls, url):
		return cls.get_file_suffix(file_url=url) in {".csv", ".txt"}

	# Selects the correct validation to run on a specific input argument
	@classmethod
	def is_valid_arg(cls, arg_type, arg_value):
		if arg_type == Argument.K:
			return cls.is_valid_k(k=arg_value)
		elif arg_type == Argument.EPSILON:
			return cls.is_valid_epsilon(epsilon=arg_value)
		elif arg_type == Argument.FILE1_URL:
			return cls.is_valid_file_url(url=arg_value)
		elif arg_type == Argument.FILE2_URL:
			return cls.is_valid_file_url(url=arg_value)
		elif arg_type == Argument.ITER:
			return cls.is_valid_iter(input_iter=arg_value)
		elif arg_type == Argument.PY_FILE:
			return True
		else:
			return False

	# Prints the appropriate error message
	@classmethod
	def print_invalid_arg_error(cls, arg_type):
		if arg_type == Argument.K:
			print(ErrorMessages.INVALID_K_ERROR_MSG.value)
		elif arg_type == Argument.EPSILON:
			print(ErrorMessages.INVALID_EPSILON_ERROR_MSG.value)
		elif arg_type == Argument.FILE1_URL:
			print(ErrorMessages.INVALID_FILE_NAME_ERROR_MSG.value)
		elif arg_type == Argument.FILE2_URL:
			print(ErrorMessages.INVALID_FILE_NAME_ERROR_MSG.value)
		elif arg_type == Argument.ITER:
			print(ErrorMessages.INVALID_ITER_ERROR_MSG.value)
		else:
			print(ErrorMessages.GENERAL_ERROR_MSG.value)

	@classmethod
	# Validates to correctness of the passed cmd command and returns errors as requested in the assignment
	def validate_cmd_arguments(cls, arguments_list):
		# Assert number of arguments passed
		if len(arguments_list) != 6:
			print(ErrorMessages.GENERAL_ERROR_MSG.value)
			sys.exit(1)

		for arg in Argument:
			if not cls.is_valid_arg(arg_type=arg, arg_value=arguments_list[arg]):
				cls.print_invalid_arg_error(arg_type=arg)
				sys.exit(1)


class DFHandler:
	"""
	The DFHandler class handles the joining of the two input datasets and defines constants according to the output DataFrame.
	The handler:
		1. Reads the input datasets and creates 2 DataFrames, one for each dataset.
		2. Sets headers for each DataFrame.
		3. Sets first column as index.
		4. Inner-joins the DataFrames.
		5. Defines constants according to the output DataFrame - isCentroid & D columns indices.
	"""
	ID_COL = ['id']
	is_centroid_col_index: int
	d_value_col_index: int

	def __init__(self, df1_url, df2_url):
		self.df1 = pd.read_csv(filepath_or_buffer=df1_url, sep=",", header=None)
		self.df2 = pd.read_csv(filepath_or_buffer=df2_url, sep=",", header=None)

	def get_joined_df(self):
		self.set_headers()
		self.set_indices()
		return self.join_dfs()

	def generate_headers(self):
		df1_header = self.ID_COL + [str(i) for i in range(1, self.df1.columns.size)]
		df2_header = self.ID_COL + [str(i) for i in range(self.df1.columns.size, self.df1.columns.size + self.df2.columns.size - 1)]

		return df1_header, df2_header

	def set_headers(self):
		self.df1.columns, self.df2.columns = self.generate_headers()

	def set_indices(self):

		# Set ID column as index
		self.df1.set_index(keys=self.ID_COL, inplace=True)
		self.df2.set_index(keys=self.ID_COL, inplace=True)

		# Cast ID from float to int
		self.df1.index = self.df1.index.astype(dtype=int, copy=False)
		self.df2.index = self.df2.index.astype(dtype=int, copy=False)

	def join_dfs(self):
		joined_df = None
		joined_df = self.df1.join(self.df2, how='inner', on=self.ID_COL, sort=True)
		# Adding D and isCentroid columns
		joined_df[IS_CENTROID_COL_NAME] = 0
		joined_df[D_VALUE_COL_NAME] = 0.0

		# Updating isCentroid and D value columns indices values
		self.is_centroid_col_index = joined_df.columns.get_loc(key=IS_CENTROID_COL_NAME)
		self.d_value_col_index = joined_df.columns.get_loc(key=D_VALUE_COL_NAME)

		return joined_df


class KmeansPPRunner:
	"""
	The KmeansPPRunner class handles the initialization and the running of the K-means++ algorithm.
	The class:
		1. Fetches the input command and verifies the correctness of its values using a reader.
		2. Initializes the DataFrame object of the dataset using a handler.
		3. Runs the K-means++ algorithm to initialize the clusters.
	"""

	reader: CommandLineReader
	handler: DFHandler

	datapoints_df: pd.DataFrame
	clusters_df: pd.DataFrame

	k: int
	iter: int
	epsilon: float
	n: int
	dimension: int

	initialized_centroids_idx_arr: list

	def __init__(self):
		# Fetch Inputs
		self.reader = CommandLineReader(cmd_input=sys.argv)

		self.k = self.reader.k
		self.iter = self.reader.iter
		self.epsilon = self.reader.epsilon

		self.handler = DFHandler(df1_url=self.reader.file1_url, df2_url=self.reader.file2_url)

		try:
			self.datapoints_df = self.handler.get_joined_df()
		except ValueError:
			self.reader.print_invalid_arg_error(arg_type=Argument.OTHER)
			exit(1)

		self.clusters_df = pd.DataFrame()
		self.n = self.datapoints_df.shape[0]
		self.dimension = self.datapoints_df.shape[1] - 2
		self.initialized_centroids_idx_arr = list()

		# Validate K with updated N
		if self.k > self.n:
			self.reader.print_invalid_arg_error(arg_type=Argument.K)
			exit(1)

		np.random.seed(NP_RANDOM_SEED)

	@staticmethod
	# Measure Euclidian distance between 2 points or from all vectors in a matrix to a vector.
	def euclidian_distance(point1, point2):
		return np.sqrt(np.sum((point1.to_numpy() - point2.to_numpy()) ** 2, axis=1))

	# Generates a random point index based on the probability function P or uniformly if first point
	def get_random_index(self):
		probability = None
		# Set random probability: if this is the first cluster, set uniform probability, else calc P
		if not self.clusters_df.empty:
			self.calc_d()
			probability = self.calc_P()
			probability.fillna(value=0.0, inplace=True)
		return int(np.random.choice(self.datapoints_df.index, p=probability))

	# Given a random index, marks the vector in that index as selected and adds it to the clusters df
	def add_new_random_centroid(self):
		# Generate random index
		new_centroid_index = self.get_random_index()
		self.initialized_centroids_idx_arr.append(new_centroid_index)

		self.datapoints_df.iloc[new_centroid_index, self.handler.is_centroid_col_index] = 1

		new_cluster = self.datapoints_df.iloc[[new_centroid_index]]
		new_cluster = pd.DataFrame(new_cluster).iloc[:, :-2]

		self.clusters_df = self.clusters_df.append(new_cluster, ignore_index=True)

	# Calculates and sets the new D values of all non-cluster vectors
	def calc_d(self):

		non_centroid_points = self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 0, self.datapoints_df.columns[:-2]]
		latest_cluster_added = self.clusters_df.tail(1)

		distances = self.euclidian_distance(point1=non_centroid_points, point2=latest_cluster_added)

		non_centroid_points_D_col = self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 0, D_VALUE_COL_NAME]

		if self.clusters_df.shape[0] > 1:
			self.datapoints_df.loc[(self.datapoints_df[IS_CENTROID_COL_NAME] == 0), [D_VALUE_COL_NAME]] = np.minimum(non_centroid_points_D_col, distances)
		else:
			# First centroid distance assignment
			self.datapoints_df.loc[(self.datapoints_df[IS_CENTROID_COL_NAME] == 0), [D_VALUE_COL_NAME]] = distances

		# set D(x) = 0 for datapoints selected as centroids.
		self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 1, [D_VALUE_COL_NAME]] = 0

	# Calculates and sets the P value of non-cluster vectors per new cluster addition
	def calc_P(self):
		D_sum = self.datapoints_df[D_VALUE_COL_NAME].sum()
		# Calc P for entire column
		return self.datapoints_df[D_VALUE_COL_NAME] / D_sum

	# Initialize centroids using K-means++ algorithm
	def initialize_centroids(self):
		for i in range(self.k):
			self.add_new_random_centroid()

		# clean the datapoints dataframe after initialization
		self.datapoints_df.drop([IS_CENTROID_COL_NAME, D_VALUE_COL_NAME], axis=1, inplace=True)
		# Now both the datapoints_df and clusters_df contains coordinates only for each point.


class KmeansRunner:
	initialized_KmeansPP_centroids: KmeansPPRunner
	final_clusters: list

	def __init__(self, centroids: KmeansPPRunner):
		self.initialized_KmeansPP_centroids = centroids
		self.final_clusters = list()

	@staticmethod
	def df_to_list_of_lists(df):
		return df.values.tolist()

	def print_output(self):
		# Print indices of the observations chosen by the K-means++ algorithm
		print(",".join(map(str, self.initialized_KmeansPP_centroids.initialized_centroids_idx_arr)))

		# Print the calculated final centroids from the K-means algorithm
		for i in range(self.initialized_KmeansPP_centroids.k):
			print(",".join(["{:.4f}".format(item) for item in self.final_clusters[i]]))

	def run_Kmeans_in_C(self):
		# Initialize centroids in Python
		self.initialized_KmeansPP_centroids.initialize_centroids()

		# Flatten data before moving to C code
		datapoints_df_list = self.df_to_list_of_lists(df=self.initialized_KmeansPP_centroids.datapoints_df)
		clusters_df_list = self.df_to_list_of_lists(df=self.initialized_KmeansPP_centroids.clusters_df)

		kmeans_c_input = (datapoints_df_list, clusters_df_list, self.initialized_KmeansPP_centroids.iter, self.initialized_KmeansPP_centroids.k, self.initialized_KmeansPP_centroids.n, self.initialized_KmeansPP_centroids.dimension)

		# Call C implementation of K-Means
		self.final_clusters = mykmeanssp.fit(*kmeans_c_input)
		self.print_output()


if __name__ == '__main__':
	try:
		# Init K-means clusters from input data using K-means++ algorithm
		KmeansPP_centroids = KmeansPPRunner()

		# Initialize a K-means runner with the generated K-means++ clusters
		runner = KmeansRunner(centroids=KmeansPP_centroids)

		# Run the K-means algorithm using C code
		runner.run_Kmeans_in_C()

	except (Exception, ValueError, OSError) as e:
		print(f"Caught error: \n\t{e}")
		print(ErrorMessages.GENERAL_ERROR_MSG.value)
		exit(1)




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


# Example of a script call: `python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt`
class Argument(IntEnum):
	# PYTHON_CALL = 0
	PY_FILE = 0
	K = 1
	ITER = 2
	EPSILON = 3
	FILE1_URL = 4
	FILE2_URL = 5


class CommandLineReader:
	def __init__(self, cmd_input):
		if len(cmd_input) == 5:
			# Assuming no `iter` provided, adding default iteration value
			cmd_input = cmd_input[Argument.PYTHON_CALL.value:Argument.K.value] + [DEFAULT_ITER] + cmd_input[
																								  Argument.K.value:]

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

	@classmethod
	# Retrieves the file URL suffix (hence type)
	def get_file_suffix(cls, file_url):
		return file_url[-4:]

	@classmethod
	# Validates that the passed URL is either a txt or csv file.
	def is_valid_file_url(cls, url):
		return cls.get_file_suffix(file_url=url) in {".csv", ".txt"}

	@classmethod
	def is_valid_arg(cls, arg_type, arg_value):
		if arg_type == Argument.K:
			return cls.is_natural(arg_value)
		elif arg_type == Argument.EPSILON:
			return arg_value.isnumeric() and int(arg_value) >= 0
		elif arg_type == Argument.FILE1_URL:
			return cls.is_valid_file_url(url=arg_value)
		elif arg_type == Argument.FILE2_URL:
			return cls.is_valid_file_url(url=arg_value)
		elif arg_type == Argument.ITER:
			return cls.is_natural(arg_value) and 1 < int(arg_value) < MAX_ITER
		elif arg_type == Argument.PY_FILE:
			return True
		else:
			return False

	@classmethod
	def print_invalid_arg_error(cls, arg_type: Argument.K):
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
			print(ErrorMessages.GENERAL_ERROR_MSG.value + "\n" + f"{arg_type.value}")

	@classmethod
	# Validates to correctness of the passed cmd command and returns errors as requested in the assignment
	def validate_cmd_arguments(cls, arguments_list):
		# Assert number of arguments passed
		if len(arguments_list) != 6:
			print(ErrorMessages.GENERAL_ERROR_MSG + " fail validate_cmd_arguments")
			sys.exit(1)

		for arg in Argument:
			if not cls.is_valid_arg(arg_type=arg, arg_value=arguments_list[arg]):
				print(arg.name)
				cls.print_invalid_arg_error(arg_type=arg)

				sys.exit(1)


class DFHandler:
	ID_COL = ['id']
	is_centroid_index: int
	d_value_index: int

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

		# print(f"df1:\n{self.df1.head()}")

	def join_dfs(self):
		joined_df = self.df1.join(self.df2, how='inner', on=self.ID_COL, sort=True)
		# Adding D and isCentroid columns
		joined_df[IS_CENTROID_COL_NAME] = 0
		joined_df[D_VALUE_COL_NAME] = 0.0

		# Updating is_centroid, D value and vector indices values
		self.is_centroid_index = joined_df.columns.get_loc(key=IS_CENTROID_COL_NAME)
		self.d_value_index = joined_df.columns.get_loc(key=D_VALUE_COL_NAME)

		# print(f"joined_df:\n{joined_df.head()}")
		return joined_df


class KmeansPPInitializer:
	def __init__(self):
		self.reader = CommandLineReader(cmd_input=sys.argv)

		self.k = self.reader.k
		self.iter = self.reader.iter
		self.epsilon = self.reader.epsilon

		self.handler = DFHandler(df1_url=self.reader.file1_url, df2_url=self.reader.file2_url)

		self.datapoints_df = self.handler.get_joined_df()
		self.clusters_df = pd.DataFrame()
		self.n = self.datapoints_df.size
		self.dimension = self.datapoints_df.shape[1] - 2
		self.initialized_centroids_idx_arr = list()

		np.random.seed(NP_RANDOM_SEED)

	@staticmethod
	# Measure Euclidian distance between 2 points or from all vectors in a matrix to a vector.
	def euclidian_distance(point1, point2):

		point1_nparray = np.array(point1)
		point2_nparray = np.array(point2)

		return np.sqrt(np.sum((point1.to_numpy() - point2.to_numpy()) ** 2, axis=1))

	# Generates a random vector index based on the probability function P or uniformly if first vector
	def get_random_index(self):
		probability = None
		# Set random probability: if this is the first cluster, set uniform probability, else calc P
		if not self.clusters_df.empty:
			print("Non-empty clusters df")
			self.calc_d()
			print("Calculated D value")

			print(f"input for calc_P:\n{self.datapoints_df}")
			probability = self.calc_P()
			probability.fillna(value=0.0, inplace=True)
			print(f"probability distribution is: {probability}")
			print("Calculated P value")
		return int(np.random.choice(self.datapoints_df.index, p=probability))

	# Given a random index, marks the vector in that index as selected and adds it to the clusters df
	def add_new_random_centroid(self):
		# Generate random index
		new_centroid_index = self.get_random_index()
		print(f"new_centroid_index: {new_centroid_index}")

		self.datapoints_df.iloc[new_centroid_index, self.handler.is_centroid_index] = 1

		print(f"datapoints_df:\n{self.datapoints_df}")

		new_cluster = self.datapoints_df.iloc[[new_centroid_index]]
		new_cluster = pd.DataFrame(new_cluster).iloc[:, :-2]

		print(f"new_cluster:\n{new_cluster}")

		self.clusters_df = self.clusters_df.append(new_cluster, ignore_index=True)
		# self.clusters_df = pd.concat(objs=[self.clusters_df, self.datapoints_df.iloc[new_centroid_index, :-2]], ignore_index=True)

	# Calculates and sets the new D values of all non-cluster vectors
	def calc_d(self):

		non_centroid_points = self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 0, self.datapoints_df.columns[:-2]]

		print(f"clusters df:\n {self.clusters_df}")
		latest_cluster_added = self.clusters_df.tail(1)

		print(f"non-centroid points:\n{non_centroid_points}")
		print(f"latest cluster added:\n{latest_cluster_added}")

		distances = self.euclidian_distance(point1=non_centroid_points, point2=latest_cluster_added)

		print(f"euclidian distance:\n{distances}")

		non_centroid_points_D_col = self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 0, D_VALUE_COL_NAME]
		print(f"non-centroid points D_col:\n{non_centroid_points_D_col}")

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

	# relevant for output
	def set_initialized_centroids_idx_arr(self):
		self.initialized_centroids_idx_arr = self.datapoints_df.index[self.datapoints_df[IS_CENTROID_COL_NAME] == 1].tolist()

	def initialize_centroids(self):
		for i in range(self.k):
			self.add_new_random_centroid()

		print("Finished initializing centroids")
		self.set_initialized_centroids_idx_arr()
		print("Finished initializing centroids idx array")

		# clean the datapoints dataframe after initialization
		self.datapoints_df.drop([IS_CENTROID_COL_NAME, D_VALUE_COL_NAME], axis=1, inplace=True)

	# now both the datapoints_df and clusters_df contains coordinates only for each point.


class KmeansPPRunner:
	final_clusters = None

	def __init__(self, initialized_Kmeans):
		self.initialized_Kmeans = initialized_Kmeans
		self.final_clusters = None

	@staticmethod
	def df_to_list_of_lists(df):
		return df.values.tolist()

	def print_output(self):
		print(",".join(map(str, self.initialized_Kmeans.initialized_centroids_idx_arr)))
		for i in range(self.initialized_Kmeans.k):
			print(",".join(["{:.4f}".format(item) for item in self.final_clusters[i]]))

	def runKmeansPP(self):
		# try:
		# Initialize centroids in Python
		self.initialized_Kmeans.initialize_centroids()

		# Call C implementation of K-Means
		print(f"Going into C code...")

		print(f"C Arguments:")
		print(f"\tdatapoints_df: {self.df_to_list_of_lists(df=self.initialized_Kmeans.datapoints_df.head(1))}\n\t\ttype: {type(self.df_to_list_of_lists(df=self.initialized_Kmeans.datapoints_df))}")
		print(f"\tclusters_df: {self.df_to_list_of_lists(df=self.initialized_Kmeans.clusters_df)}\n\t\ttype: {type(self.df_to_list_of_lists(df=self.initialized_Kmeans.clusters_df))}")
		print(f"\titer: {self.initialized_Kmeans.iter}")
		print(f"\tn: {self.initialized_Kmeans.n}")
		print(f"\tdimension: {self.initialized_Kmeans.dimension}")

		datapoints_df_list = self.df_to_list_of_lists(df=self.initialized_Kmeans.datapoints_df)
		clusters_df_list = self.df_to_list_of_lists(df=self.initialized_Kmeans.clusters_df)

		# input_tuple = (datapoints_df_list,
		# 			   clusters_df_list,
		# 			   self.initialized_Kmeans.iter,
		# 			   self.initialized_Kmeans.k,
		# 			   self.initialized_Kmeans.n,
		# 			   self.initialized_Kmeans.dimension)

		my_final_clusters = mykmeanssp.python_fit(datapoints_df_list,
													clusters_df_list,
													self.initialized_Kmeans.iter,
													self.initialized_Kmeans.k,
													self.initialized_Kmeans.n,
													self.initialized_Kmeans.dimension)
		# print(f"input tuple:\n{input_tuple}")
		# self.final_clusters = mykmeanssp.run_kmeans(input_tuple)
		print(f"Finished running C code!")
		self.final_clusters = my_final_clusters
		self.print_output()

	# except Exception as e:
	#     print(self.GENERAL_ERROR_MSG) # TODO: ask tomer
	#    print(e)
	#    return


if __name__ == '__main__':
	print(f"command line arguments passed: {sys.argv}")
	initialized_kmeans = KmeansPPInitializer()
	runner = KmeansPPRunner(initialized_Kmeans=initialized_kmeans)
	runner.runKmeansPP()

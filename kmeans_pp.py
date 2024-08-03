import array
import sys
from enum import IntEnum, StrEnum
import pandas as pd
import numpy as np
import mykmeanssp

MAX_ITER = 1000
DEFAULT_ITER = 200
EPSILON = 0.001

NP_RANDOM_SEED = 1234

IS_CENTROID_COL_NAME = 'isCentroid'
D_VALUE_COL_NAME = 'D'


class ErrorMessages(StrEnum):
	INVALID_K_ERROR_MSG = "Invalid number of clusters!"
	INVALID_ITER_ERROR_MSG = "Invalid maximum iteration!"
	INVALID_FILE_NAME_ERROR_MSG = "Invalid file names!"
	INVALID_EPSILON_ERROR_MSG = "Invalid epsilon!"
	GENERAL_ERROR_MSG = "An Error Has Occurred"


# Example of a script call: `python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt`
class Argument(IntEnum):
	PYTHON_CALL = 0
	PY_FILE = 1
	K = 2
	ITER = 3
	EPSILON = 4
	FILE1_URL = 5
	FILE2_URL = 6


class CommandLineReader:
	file1_url: str
	file2_url: str
	n: int
	k: int
	iter: int
	epsilon: float

	def __init__(self, cmd_input: list):
		if len(cmd_input) == 5:
			# Assuming no `iter` provided, adding default iteration value
			cmd_input = cmd_input[Argument.PYTHON_CALL.value:Argument.K.value] + [DEFAULT_ITER] + cmd_input[Argument.K.value:]

		self.validate_cmd_arguments(arguments_list=cmd_input)

		self.k = int(cmd_input[Argument.K.value])
		self.iter = int(cmd_input[Argument.ITER.value])
		self.epsilon = float(cmd_input[Argument.EPSILON.value])
		self.file1_url = str(cmd_input[Argument.FILE1_URL.value])
		self.file2_url = str(cmd_input[Argument.FILE2_URL.value])

	@staticmethod
	# Checks whether the string `n` represents a natural number(0 excluded)
	def is_natural(n) -> bool:
		return str(n).isdigit() and float(n) == int(n) and int(n) > 0

	@classmethod
	# Retrieves the file URL suffix (hence type)
	def get_file_suffix(cls, file_url: str) -> str:
		return file_url.split(".")[-1]

	@classmethod
	# Validates that the passed URL is either a txt or csv file.
	def is_valid_file_url(cls, url: str) -> bool:
		return cls.get_file_suffix(file_url=url) in {".csv", ".txt"}

	@classmethod
	def is_valid_arg(cls, arg_type: Argument, arg_value: str) -> bool:
		match arg_type:
			case Argument.K:
				return cls.is_natural(arg_value)
			case Argument.EPSILON:
				return arg_value.isnumeric() and int(arg_value) >= 0
			case Argument.FILE1_URL:
				return cls.is_valid_file_url(url=arg_value)
			case Argument.FILE2_URL:
				return cls.is_valid_file_url(url=arg_value)
			case Argument.ITER:
				cls.is_natural(arg_value) and 1 < int(arg_value) < MAX_ITER
			case _:
				return False

	@classmethod
	def print_invalid_arg_error(cls, arg_type: Argument):
		match arg_type:
			case Argument.K:
				print(ErrorMessages.INVALID_K_ERROR_MSG.value)
			case Argument.EPSILON:
				print(ErrorMessages.INVALID_EPSILON_ERROR_MSG.value)
			case Argument.FILE1_URL:
				print(ErrorMessages.INVALID_FILE_NAME_ERROR_MSG.value)
			case Argument.FILE2_URL:
				print(ErrorMessages.INVALID_FILE_NAME_ERROR_MSG.value)
			case Argument.ITER:
				print(ErrorMessages.INVALID_ITER_ERROR_MSG.value)
			case _:
				print(ErrorMessages.GENERAL_ERROR_MSG.value)

	@classmethod
	# Validates to correctness of the passed cmd command and returns errors as requested in the assignment
	def validate_cmd_arguments(cls, arguments_list: list) -> None:
		# Assert number of arguments passed
		if len(arguments_list) != 6:
			print(ErrorMessages.GENERAL_ERROR_MSG)
			sys.exit(1)

		for arg in Argument:
			if not cls.is_valid_arg(arg_type=arg, arg_value=arguments_list[arg]):
				cls.print_invalid_arg_error(arg_type=arg)
				sys.exit(1)


class DFHandler:
	ID_COL = ['id']
	df1: pd.DataFrame
	df2: pd.DataFrame

	def __init__(self, df1_url: str, df2_url: str):
		self.df1 = pd.read_csv(filepath_or_buffer=df1_url, sep=",", header=None)
		self.df2 = pd.read_csv(filepath_or_buffer=df2_url, sep=",", header=None)

	def get_joined_df(self) -> pd.DataFrame:
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
		self.df1.set_index(keys=self.ID_COL, inplace=True)
		self.df2.set_index(keys=self.ID_COL, inplace=True)

	def join_dfs(self) -> pd.DataFrame:
		joined_df = self.df1.join(self.df2, how='inner', on=str(self.ID_COL), sort=True)
		joined_df.insert(loc=len(joined_df.columns), column=D_VALUE_COL_NAME, value=np.zeros(joined_df.shape[1]))
		joined_df.insert(loc=len(joined_df.columns), column=IS_CENTROID_COL_NAME, value=np.zeros(joined_df.shape[1]))
		return joined_df


class KmeansPPInitializer:
	reader: CommandLineReader
	handler: DFHandler
	datapoints_df: pd.DataFrame
	clusters_df: pd.DataFrame
	n: int
	k: int
	dimension: int
	iter: int
	epsilon: float
	initialized_centroids_idx_arr: list

	def __init__(self):
		self.reader = CommandLineReader(cmd_input=sys.argv)

		self.k = self.reader.k
		self.iter = self.reader.iter
		self.epsilon = self.reader.epsilon

		self.handler = DFHandler(df1_url=self.reader.file1_url, df2_url=self.reader.file2_url)

		self.datapoints_df = self.handler.get_joined_df()
		self.clusters_df = pd.DataFrame()
		self.n = self.datapoints_df.size
		self.dimension = self.datapoints_df.shape[1]
		self.initialized_centroids_idx_arr = list()

		np.random.seed(NP_RANDOM_SEED)

	@staticmethod
	# Measure Euclidian distance between 2 points or from all vectors in a matrix to a vector.
	def euclidian_distance(point1: np.ndarray | pd.DataFrame, point2: np.ndarray | pd.DataFrame) -> np.ndarray | float:
		return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))

	# Generates a random vector index based on the probability function P or uniformly if first vector
	def get_random_index(self) -> int:
		probability = None
		# Set random probability: if this is the first cluster, set uniform probability, else calc P
		if not self.clusters_df.empty:
			self.calc_d()
			probability = self.calc_P()
		return np.random.choice(self.datapoints_df.index, p=probability)

	# Given a random index, marks the vector in that index as selected and adds it to the clusters df
	def add_new_random_centroid(self) -> None:
		new_centroid_index = self.get_random_index()
		self.datapoints_df.iloc[new_centroid_index, [IS_CENTROID_COL_NAME]] = 1
		self.clusters_df = pd.concat(objs=[self.clusters_df, self.datapoints_df.iloc[new_centroid_index]], ignore_index=True, axis=0)

	# Calculates and sets the new D values of all non-cluster vectors
	def calc_d(self):
		distances = self.euclidian_distance(point1=self.datapoints_df[self.datapoints_df[IS_CENTROID_COL_NAME == 0]], point2=self.clusters_df.tail(1))
		self.datapoints_df.where(cond=self.datapoints_df[D_VALUE_COL_NAME] <= distances, other=distances, inplace=True)
		# set D(x) = 0 for datapoints selected as centroids.
		self.datapoints_df.loc[self.datapoints_df[IS_CENTROID_COL_NAME] == 1, [D_VALUE_COL_NAME]] = 0
	
	# Calculates and sets the P value of non-cluster vectors per new cluster addition
	def calc_P(self) -> np.array:
		D_sum = self.datapoints_df[D_VALUE_COL_NAME].sum()
		# Calc P for entire column
		return self.datapoints_df.loc[D_VALUE_COL_NAME] / D_sum
	
	# relevant for output
	def set_initialized_centroids_idx_arr(self):
		self.initialized_centroids_idx_arr = self.datapoints_df.index[self.datapoints_df[IS_CENTROID_COL_NAME] == 1].tolist()

	def initialize_centroids(self):
		for i in range(self.k):
			self.add_new_random_centroid()
		
		self.set_initialized_centroids_idx_arr()
		# clean the datapoints dataframe after initialization
		self.datapoints_df.drop([IS_CENTROID_COL_NAME, D_VALUE_COL_NAME], axis=1, inplace=True)

		# now both the datapoints_df and clusters_df contains coordinates only for each point. 
	

class KmeansPPRunner:
	initialized_Kmeans: KmeansPPInitializer
	final_clusters: list[list]

	def __init__(self, initialized_Kmeans: KmeansPPInitializer):
		self.initialized_Kmeans = initialized_Kmeans

	@staticmethod
	def df_to_list_of_lists(df):
		return df.values.to_list()
	
	def print_output(self):
		print(",".join(map(str, self.initialized_Kmeans.initialized_centroids_idx_arr)))
		for i in range(self.initialized_Kmeans.k):
			print(",".join([format(item, '.4f') for item in self.final_clusters[i]]))
	
	def runKmeansPP(self):
		# try:
		self.initialized_Kmeans.initialize_centroids()
		self.final_clusters = mykmeanssp.run_kmeans(self.initialized_Kmeans.datapoints_df, self.initialized_Kmeans.clusters_df, self.initialized_Kmeans.iter, self.initialized_Kmeans.k, self.initialized_Kmeans.n, self.initialized_Kmeans.dimension)
		self.print_output()
		
		# except Exception as e:
		# 	print(self.GENERAL_ERROR_MSG) # TODO: ask tomer
		#	print(e)
		#	return


if __name__ == '__main__':
	initialized_kmeans = KmeansPPInitializer()
	runner = KmeansPPRunner(initialized_Kmeans=initialized_kmeans)
	runner.runKmeansPP()



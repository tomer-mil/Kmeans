import sys
from enum import IntEnum, StrEnum
import pandas as pd
import numpy as np

MAX_ITER = 1000
DEFAULT_ITER = 200
EPSILON = 0.001

NP_RANDOM_SEED = 1234

WAS_SELECTED_AS_CENTROID_COL_NAME = 'wasSelectedAsCentroid'
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
			cmd_input = cmd_input[Argument.PYTHON_CALL.value:Argument.K.value] + [DEFAULT_ITER] + cmd_input[
																								  Argument.K.value:]

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
		joined_df.insert(loc=len(joined_df.columns), column=WAS_SELECTED_AS_CENTROID_COL_NAME, value=np.zeros(joined_df.shape[1]))
		return joined_df


class KmeansPPInitializer:
	reader: CommandLineReader
	handler: DFHandler
	datapoints_df: pd.DataFrame
	clusters_df: pd.DataFrame
	n: int
	k: int
	iter: int
	epsilon: float

	def __init__(self):
		self.reader = CommandLineReader(cmd_input=sys.argv)

		self.k = self.reader.k
		self.iter = self.reader.iter
		self.epsilon = self.reader.epsilon

		self.handler = DFHandler(df1_url=self.reader.file1_url, df2_url=self.reader.file2_url)

		self.datapoints_df = self.handler.get_joined_df()
		self.clusters_df = pd.DataFrame()
		self.n = self.datapoints_df.size

		np.random.seed(NP_RANDOM_SEED)

	@staticmethod
	# Measure Euclidian distance between 2 points or from all vectors in a matrix to a vector.
	def euclidian_distance(point1: np.ndarray | pd.DataFrame, point2: np.ndarray | pd.DataFrame) -> np.ndarray | float:
		return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))

	# Generates a random vector index based on the probability function P or uniformly if first vector
	def get_random_index(self) -> int:
		# Set random probability: if this is the first cluster, set uniform probability, else calc P
		probability = self.calc_P() if not self.clusters_df.shape[0] == 0 else None

		return np.random.choice(self.datapoints_df[self.datapoints_df[WAS_SELECTED_AS_CENTROID_COL_NAME] == 0].index, p=probability)

	# Given a random index, marks the vector in that index as selected and adds it to the clusters df
	def add_new_random_centroid(self) -> None:
		new_centroid_index = self.get_random_index()
		self.datapoints_df[new_centroid_index][WAS_SELECTED_AS_CENTROID_COL_NAME] = 1
		self.clusters_df = pd.concat(objs=[self.clusters_df, self.datapoints_df.loc[new_centroid_index]], ignore_index=True, axis=0)

	# Calculates and sets the new D values of all non-cluster vectors
	def calc_d(self):
		distances = self.euclidian_distance(point1=self.datapoints_df[self.datapoints_df[WAS_SELECTED_AS_CENTROID_COL_NAME == 0]], point2=self.clusters_df[-1])
		self.datapoints_df.where(cond=self.datapoints_df[D_VALUE_COL_NAME] <= distances, other=distances, inplace=True)

	# Calculates and sets the P value of non-cluster vectors per new cluster addition
	def calc_P(self):

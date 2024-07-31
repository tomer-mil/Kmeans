import sys
import pandas as pd
import numpy  as np
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

def generate_datapoints_df(file1_url: str, file2_url: str) -> pd.DataFrame:
    file1_df = pd.read_csv(filepath_or_buffer=file1_url, sep=",", header=None)
    file2_df = pd.read_csv(filepath_or_buffer=file2_url, sep=",", header=None)

    datapoints_df = file1_df.join(file2_df, how="inner", sort=True)

    return datapoints_df

def init_centroids(datapoints_df: pd.DataFrame) -> pd.DataFrame:
    centroids_df = pd.DataFrame()
    

def calc_d(datapoints_df: pd.DataFrame) -> None:
    
    pass

datapoints_df = generate_datapoints_df(file1_url=sys.argv[4], file2_url=sys.argv[5])
cluster_df = pd.DataFrame()



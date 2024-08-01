import pandas as pd

ID_COL = ['id']

file1_url = "/Users/tomermildworth/Documents/University/Year_4/Semester_b/Software_Project/HW2/tests/input_1_db_1.txt"
file2_url = "/Users/tomermildworth/Documents/University/Year_4/Semester_b/Software_Project/HW2/tests/input_1_db_2.txt"

file1_df = pd.read_csv(filepath_or_buffer=file1_url, sep=",", header=None)
file2_df = pd.read_csv(filepath_or_buffer=file2_url, sep=",", header=None)

# file1_col_names = ['id'] + [str(i) for i in range(1, len(file1_df.columns) + 1)]
# file2_col_names = ['id'] + [str(i) for i in range(len(file1_df.columns) + 1, len(file1_df.columns) + 1 + len(file2_df.columns))]

file1_df.columns = ID_COL + [str(i) for i in range(1, len(file1_df.columns))]
file2_df.columns = ID_COL + [str(i) for i in range(len(file1_df.columns), len(file1_df.columns) + len(file2_df.columns) - 1)]

file1_df['id'] = file1_df['id'].apply(int)
file2_df['id'] = file2_df['id'].apply(int)

file1_df.set_index('id', inplace=True)
file2_df.set_index('id', inplace=True)

# print(len(file1_df.columns))
# print(file1_col_names)
# print(file2_col_names)

print(f"df1: \n{file1_df.head()}\n")
print(f"df2: \n{file2_df.head()}\n")

new_df = file1_df.join(file2_df, how="inner", on='id', sort=True)
# new_df['id'] = new_df['id'].apply(int)

print(f"joined df: \n{new_df.head()}\n")
print(f"joined df index: {new_df.index}")

test_df = pd.DataFrame(columns=['x', 'y'])
print(test_df.shape)


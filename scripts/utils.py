#%%
# Imports
import pandas as pd
import os
#%%
def convert_parquet_to_csv(directory):
    try:
        # Create a new directory for the CSV files
        csv_directory = os.path.join(directory, 'csv')
        os.makedirs(csv_directory, exist_ok=True)

        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(directory, filename))
                # Save the CSV files in the new directory
                df.to_csv(os.path.join(csv_directory, filename[:-8] + '.csv'), index=False)
        print("All parquet files have been converted to CSV.")
    except Exception as e:
        print(f"Error: {e}")
# %%
def split_column(path, column_name, keyword, file_path):
    try:
        df = pd.read_csv(path)
        df[['question', 'response']] = df[column_name].str.split(keyword, expand=True)
        df.drop(column_name, axis=1, inplace=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error: {e}")
        return None
#%%
def remove_string(path, column_name, string_to_remove, file_path):
    try:
        df = pd.read_csv(path)
        # Check if column_name is a valid column in df
        if column_name not in df.columns:
            print(f"Error: {column_name} is not a valid column in the DataFrame")
            return None

        # Check if the column contains None values
        if df[column_name].isnull().any():
            print(f"Error: {column_name} contains None values")
            return None

        df[column_name] = df[column_name].str.replace(string_to_remove, '')
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# %%
def concat_csv(file_path1, file_path2, output_file_path):
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        df = pd.concat([df1, df2])
        df.to_csv(output_file_path, index=False)
        print(f"CSV files successfully concatenated and saved to {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")

# %%
def drop_col(file_path, col_name):
    try:
        df = pd.read_csv(file_path)
        df.drop(col_name, axis=1, inplace=True)
        df.to_csv(file_path, index=False)
        print(f"Column {col_name} successfully dropped from {file_path}")
    except Exception as e:
        print(f"Error: {e}")
# %%
def remove_lines(path, col_name, string):
    try:
        df = pd.read_csv(path)
        filtered_df = df[df[col_name].str.contains("\[/INST\]", na=False)] 
        filtered_df.to_csv(path, index=False)
        print(f"Lines containing {string} successfully removed from {path}")
    except Exception as e:
        print(f"Error: {e}")
# %%

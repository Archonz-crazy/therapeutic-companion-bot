#%%
# Imports
import pandas as pd
import os

#%%
def convert_parquet_to_csv(directory):
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(directory, filename))
                df.to_csv(os.path.join(directory, filename[:-8] + '.csv'), index=False)
        print("All parquet files have been converted to CSV.")
    except Exception as e:
        print(f"Error: {e}")
# %%
convert_parquet_to_csv("/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files")
# %%
def split_column(df, column_name, keyword, file_path):
    """
    This function splits a column in a DataFrame into two at a certain keyword,
    deletes the original column, and saves the DataFrame back to the same CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame
    column_name (str): The name of the column to split
    keyword (str): The keyword at which to split the column
    file_path (str): The path of the CSV file

    Returns:
    None
    """
    try:
        df[[f'{column_name}_1', f'{column_name}_2']] = df[column_name].str.split(keyword, expand=True)
        df = df.drop(columns=[column_name])
        df.to_csv(file_path, index=False)
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error: {e}")
# %%
df = pd.read_csv("/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files/train-00000-of-00001-01391a60ef5c00d9.csv")
split_column(df, "text", "<ASSISTANT>", "/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files/train-00000-of-00001-01391a60ef5c00d9.csv")
# %%
df.head(5)
# %%

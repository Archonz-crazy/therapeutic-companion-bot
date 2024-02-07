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
def split_column(df, column_name, keyword, file_path):
    try:
        df[[f'{column_name}_1', f'{column_name}_2']] = df[column_name].str.split(keyword, expand=True)
        df = df.drop(columns=[column_name])
        df.to_csv(file_path, index=False)
        print(f"DataFrame successfully saved to {file_path}")
        return df
    except Exception as e:
        print(f"Error: {e}")

# %%
def remove_string(df, column_name, string_to_remove, file_path):
    try:
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

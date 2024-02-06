#%%
import pandas as pd
import pyarrow.parquet as pq
#%%
def parquet_handle(path):
    df = pd.read_parquet(path)
    # Display the first few rows
    print(df.iloc[0])

parquet_handle(r"C:\Users\1ga17\OneDrive\Desktop\data_capstone_1\parquet_files\train-00000-of-00001.parquet")

# %%

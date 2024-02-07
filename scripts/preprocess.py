#%%
# Imports
import pandas as pd

#%%
# Converting all the parquet files to CSV
from utils import convert_parquet_to_csv
from utils import split_column
from utils import remove_string
from utils import concat_csv

convert_parquet_to_csv("/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files")
# %%
# Now that we have all the CSV files, we preprocess them according to their specifications

# Concatenating p1 and p2 as they are similar
concat_csv("../data/parquet_files/csv/p1.csv","../data/parquet_files/csv/p2.csv", "../data/parquet_files/csv/p1_p2.csv")
# %%
# Splitting p5 into two columns
df = pd.read_csv("/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files/csv/p5.csv")
#%%
df = split_column(df, "text", "<ASSISTANT>: ", "../data/parquet_files/csv/p5.csv")

# %%
remove_string(df, "text_1", "<HUMAN>: ", "../data/parquet_files/csv/p5.csv")
# %%

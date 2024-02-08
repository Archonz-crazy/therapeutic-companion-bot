#%%
# Imports
import pandas as pd
from utils import convert_parquet_to_csv
from utils import split_column
from utils import remove_string
from utils import concat_csv
from utils import drop_col
from utils import remove_lines

#%%
# Converting all the parquet files to CSV
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
remove_string(df, "question", "<HUMAN>: ", "../data/parquet_files/csv/p5.csv")
   # %%
# processing p3 and p6 because they are similar
# First, dropping transcript column from p6
drop_col("../data/parquet_files/csv/p6.csv", "transcript_id")
# %%
# Then concatenating p3 and p6
concat_csv("../data/parquet_files/csv/p3.csv", "../data/parquet_files/csv/p6.csv", "../data/parquet_files/csv/p3_p6.csv")
# %%
# Now processing p7
df = pd.read_csv("/Users/mahi/Documents/therapeutic-companion-bot/data/parquet_files/csv/p7.csv")

remove_string(df, "text", "<s>[INST] ", "../data/parquet_files/csv/p7.csv")

#%%
df = split_column(df, "text", "[/INST]", "../data/parquet_files/csv/p7.csv")

# %%

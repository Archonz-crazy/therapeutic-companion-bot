#%%
# Imports
import pandas as pd
from utils import convert_parquet_to_csv
from utils import split_column
from utils import remove_string
from utils import concat_csv
from utils import drop_col
from utils import remove_lines
from utils import rename_col
from utils import concat_csv_list
#%%
# Converting all the parquet files to CSV
convert_parquet_to_csv("../data/parquet_files")
# %%
# Now that we have all the CSV files, we preprocess them according to their specifications

# Concatenating p1 and p2 as they are similar
concat_csv("../data/parquet_files/csv/p1.csv","../data/parquet_files/csv/p2.csv", "../data/parquet_files/csv/p1_p2.csv")

#%%
# As p3 and p6 are similar, we will concatenate them
drop_col("../data/parquet_files/csv/p6.csv", "transcript_id")
concat_csv("../data/parquet_files/csv/p3.csv", "../data/parquet_files/csv/p6.csv", "../data/parquet_files/csv/p3_p6.csv")

#%%
# No preprocessing required for p4 as of now

#%%
# Splitting p5 into question and response and removing the unwanted string
split_column("text", "<ASSISTANT>:", "../data/parquet_files/csv/p5.csv")
remove_string("../data/parquet_files/csv/p5.csv", "Question", "<HUMAN>: ")

# %%
# removing the string from p7 and splitting it into question and response
remove_string("../data/parquet_files/csv/p7.csv", "text", "<s>[INST] ")
remove_lines("../data/parquet_files/csv/p7.csv","text", "[/INST] ")
split_column("text", "\[\/INST\] ", "../data/parquet_files/csv/p7.csv")
# %%
# Removing the unwanted strings from p8
remove_string("../data/parquet_files/csv/p8.csv", "instruction", "### Instruction:")
remove_string("../data/parquet_files/csv/p8.csv", "instruction", "### Response:")

 # %%
#Concat p5 and p7 csv files
concat_csv("../data/parquet_files/csv/p5.csv","../data/parquet_files/csv/p7.csv", "../data/parquet_files/csv/p5_p7.csv")
# %%
#rename_col("../data/parquet_files/csv/p8.csv", "instruction", "Question")
#rename_col("../data/parquet_files/csv/p8.csv", "output", "Response")

# %%
rename_col("../data/parquet_files/csv/p8.csv", "Question", "Response")

# %%
#Concat p5_p7, and p8 csv files
concat_csv("../data/parquet_files/csv/p5_p7.csv","../data/parquet_files/csv/p8.csv", "../data/parquet_files/csv/p5_p7_p8.csv")

# %%
#Parquet Files done

# %%
#Start with CSV
concat_csv_list(['../data/TypesDisease/adhd.csv', '../data/TypesDisease/aspergers.csv', 
                    '../data/TypesDisease/depression.csv', '../data/TypesDisease/ocd.csv', 
                    '../data/TypesDisease/ptsd.csv'], "../data/TypesDisease/")


# %%

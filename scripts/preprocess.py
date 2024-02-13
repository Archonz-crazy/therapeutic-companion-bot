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
from utils import drop_columns
from utils import json_to_csv
from utils import doc_to_text
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
rename_col("../data/parquet_files/csv/p8.csv", "instruction", "Question")
rename_col("../data/parquet_files/csv/p8.csv", "output", "Response")

# %%
#rename_col("../data/parquet_files/csv/p8.csv", "Question", "Response")

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

#%%
# Drop columns which felt unnecessary
columns_to_drop = ['author', 'created_utc', 'id', 'num_comments', 'score', 'upvote_ratio', 'url']
drop_col("../data/TypesDisease/all_types.csv", columns_to_drop)
# %%
# Dropping some columns from the mental_disorders_reddit.csv
columns_to_drop = ['created_utc', "over_18"]
drop_col("../data/TypesDisease/mental_disorders_reddit.csv", columns_to_drop)
# %%
# Reoroderingand renaming all_types.csv
rename_col("../data/TypesDisease/all_types.csv", "body", "selftext")
df = pd.read_csv("../data/TypesDisease/all_types.csv")
df = df[['title', 'selftext', 'subreddit']]
df.to_csv("../data/TypesDisease/all_types.csv", index=False)
# %%
#concating all_types csv with mental_disorders_reddit csv file
concat_csv("../data/TypesDisease/all_types.csv", "../data/TypesDisease/mental_disorders_reddit.csv", "../data/TypesDisease/all_types.csv")
# %%
#renaming mental_health_FAQ csv file's column name
rename_col("../data/QA_sample/mental_health_conversation - Mental_Health_FAQ.csv", "Answers", "Response")
rename_col("../data/QA_sample/mental_health_conversation - Mental_Health_FAQ.csv", "Questions", "Question")

# %%
#concating the QA sample mental health conversation csv file with p5_p7_p8 file
concat_csv("../data/QA_sample/mental_health_conversation - Mental_Health_FAQ.csv", "../data/parquet_files/csv/p5_p7_p8.csv", "../data/parquet_files/csv/p5_p7_p8.csv")

# %%
#convert json files to csv
# Read the JSON file
json_to_csv("../data/json")


# %%
# now all json files are converted to csv, so we will rename few columns, to concat with other csv file
rename_col("../data/json/csv/combined_data.csv", "Context", "Question")
# %%
#concat with parquet_csv
concat_csv("../data/json/csv/combined_data.csv", "../data/parquet_files/csv/p5_p7_p8.csv", "../data/parquet_files/csv/p5_p7_p8_combined_json.csv")
# %%
#Now we will convert the word file to text
doc_to_text("../data/who_and_corpus/")

# %%
from docx import Document

def dtt(file_path):
    # Load the .docx file
    doc = Document(file_path)
    full_text = []
    
    # Iterate through each paragraph in the document and append its text to the list
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    # Join all paragraphs text into a single string
    return '\n'.join(full_text)

# Example usage
file_path = '../data/who_and_corpus/suicide_prevention.docx'  # Replace this with the path to your .docx file
text = dtt(file_path)
print(text)
# %%


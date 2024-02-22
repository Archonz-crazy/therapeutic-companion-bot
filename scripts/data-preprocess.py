#%%
# this file includes data preprocessing using NLTK, spacy and Gensim
#importing libraries
import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
from utils import remove_special_characters, remove_stopwords_and_lemmatize

#%%
# Read text files with .txt extension in "../data/knowledge" directory
directory = "../data/knowledge"
output_file_path = os.path.join(directory, "text_preprocess.txt")

# Ensure the output file is empty before starting
with open(output_file_path, 'w') as file:
    file.write("")

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            text = file.read().lower()  # Convert to lower case                 # Remove spaces, special characters, and empty lines
            text = remove_special_characters(text)
            # Remove stopwords and lemmatize it
            text = remove_stopwords_and_lemmatize(text)
            # Append the processed tokens to the single output file
            with open(output_file_path, 'a') as outfile:
                outfile.write(" ".join(text) + "\n")
print("preprocessing done \n")
with open(output_file_path, 'r') as file:
    contents = file.read()
    print(contents)
#%%
# TRYING TO USE CHUNK SIZE TO PROCESS THE DATA IN BATCH
# Function to preprocess all text columns in a DataFrame chunk
def preprocess_text_columns(chunk):
    # Identify text columns (assuming columns with object dtype are text columns)
    text_columns = chunk.select_dtypes(include=['object']).columns
    
    # Apply preprocessing to each text column
    for column in text_columns:
        chunk[column] = chunk[column].astype(str)
        chunk[column] = chunk[column].apply(remove_special_characters)  # Remove special characters
        chunk[column] = chunk[column].apply(lambda x: x.replace("NaN", ""))  # Remove NaN values
        chunk[column] = chunk[column].apply(remove_stopwords)  # Remove stopwords
        chunk[column] = chunk[column].apply(lemmatize_text)  # Lemmatize text
    return chunk

#%%
# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename == "labelled.csv":
        file_path = os.path.join(directory, filename)
        chunk_size = 80000  # Define the size of each chunk
        
        # Process each chunk within the file
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = preprocess_text_columns(chunk)
            # Here, you can further analyze, save, or aggregate the processed data as needed
            
            # Example: Save the processed chunk (consider appending if the file is large)
            processed_chunk.to_csv(file_path.replace('.csv', '_processed.csv'), mode='a', index=False, header=not os.path.exists(file_path.replace('.csv', '_processed.csv')))


# %%

#%%
# this file includes data preprocessing using NLTK, spacy and Gensim
#importing libraries
import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
from utils import remove_special_characters, remove_stopwords, lemmatize_text

#%%
# Read text files with .txt extension in "../data/knowledge" directory
directory = "../data/knowledge"
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            # Remove spaces, special characters, and empty lines
            text = remove_special_characters(text)
            # Remove stopwords
            text = remove_stopwords(text)
            # Lemmatize text
            text = lemmatize_text(text)
            # Perform further processing on the preprocessed text
            print(text)


#%%
# TRYING TO USE CHUNK SIZE TO PROCESS THE DATA IN BATCH
# Function to preprocess all text columns in a DataFrame chunk
def preprocess_text_columns(chunk):
    # Identify text columns (assuming columns with object dtype are text columns)
    text_columns = chunk.select_dtypes(include=['object']).columns
    
    # Apply preprocessing to each text column
    for column in text_columns:
        chunk[column] = chunk[column].astype(str)
        chunk[column] = chunk[column].apply(lambda x: x.replace(" ", ""))  # Remove spaces
        chunk[column] = chunk[column].apply(remove_special_characters)  # Remove special characters
        chunk[column] = chunk[column].apply(lambda x: x.replace("NaN", ""))  # Remove NaN values
        chunk[column] = chunk[column].apply(remove_stopwords)  # Remove stopwords
        chunk[column] = chunk[column].apply(lemmatize_text)  # Lemmatize text
    return chunk

#%%
# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        chunk_size = 80000  # Define the size of each chunk
        
        # Process each chunk within the file
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = preprocess_text_columns(chunk)
            # Here, you can further analyze, save, or aggregate the processed data as needed
            
            # Example: Save the processed chunk (consider appending if the file is large)
            processed_chunk.to_csv(file_path.replace('.csv', '_processed.csv'), mode='a', index=False, header=not os.path.exists(file_path.replace('.csv', '_processed.csv')))


# %%

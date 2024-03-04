#%%
# this file includes data preprocessing using NLTK, spacy and Gensim
#importing libraries
import os
import pandas as pd
from utils import remove_special_characters, remove_stopwords_and_lemmatize

#%%
# Read text files with .txt extension in "../data/knowledge" directory
directory = "../data/knowledge"
# Ensure the output file is empty before starting
output_file_path = os.path.join(directory, "text_preprocess.txt")
with open(output_file_path, 'w') as file:
    file.write("")

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            text = file.read().lower()   # Convert to lower case
            # Remove spaces, special characters, and empty lines
            text = remove_special_characters(text)
            # Remove stopwords and lemmatize it
            text = remove_stopwords_and_lemmatize(text)
            #print("text file before joining",text)
            # Append the processed tokens to the single output file
            with open(output_file_path, 'a') as outfile:
                outfile.write(" ".join(text) + "\n")
'''
print("preprocessing done \n")
with open(output_file_path, 'r') as file:
    contents = file.read()
    print(contents)
'''

<<<<<<< HEAD
# %%

def preprocess_text(text):
    text = str(text).lower()  # Convert to lower case
    text = remove_special_characters(text)  # Remove special characters
    text = text.replace("NaN", "")
    text = text.replace("nan", "")# Rejoin words and remove "NaN"
    text = remove_stopwords_and_lemmatize(text)  # Process each word
    return text

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    text_columns = df.select_dtypes(include=['object']).columns
    
    for column in text_columns:
        df[column] = df[column].apply(preprocess_text)
    
    processed_file_path = file_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_file_path, index=False)
    print(f"Preprocessed data saved to: {processed_file_path}")
    print(df.head())

#%%
# Process each CSV file in the directory
directory = "../data/knowledge"
for filename in os.listdir(directory):
    if filename == "all_types.csv":
        file_path = os.path.join(directory, filename)
        preprocess_csv(file_path)

# %%
# Preprocess each csv file in the sample_q_and_a directory
directory = "../data/sample_q_and_a"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        preprocess_csv(file_path)
=======
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
        chunk_size = 10000  # Define the size of each chunk
        
        # Process each chunk within the file
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = preprocess_text_columns(chunk)
            # Here, you can further analyze, save, or aggregate the processed data as needed
            
            # Example: Save the processed chunk (consider appending if the file is large)
            processed_chunk.to_csv(file_path.replace('.csv', '_processed.csv'), mode='a', index=False, header=not os.path.exists(file_path.replace('.csv', '_processed.csv')))
>>>>>>> main

#%%
        

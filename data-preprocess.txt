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

# %%

def preprocess_text(text):
    text = str(text).lower()  # Convert to lower case
    text = remove_special_characters(text)  # Remove special characters
    text = text.replace("NaN", "")  # Rejoin words and remove "NaN"
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
# %%

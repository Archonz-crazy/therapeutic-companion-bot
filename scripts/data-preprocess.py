#%%
# this file includes data preprocessing using NLTK, spacy and Gensim
#importing libraries
import os
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

# %%
# Read all csv files with .csv extension from the knowledge directory
# and preprocess the text using NLTK, spacy, or Gensim
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        # Preprocess the text in the dataframe
        with open(file_path, 'r') as file:
            text = file.read()
            # Remove spaces, special characters, and NaN values from csv files
            text = text.replace(" ", "")  # Remove spaces
            text = remove_special_characters(text)  # Remove special characters
            text = text.replace("NaN", "")  # Remove NaN values
            # Remove stopwords
            text = remove_stopwords(text)
            # Lemmatize text
            text = lemmatize_text(text)
            print(text)

# %%

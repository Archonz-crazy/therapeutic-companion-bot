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
output_dir = os.path.join(directory, 'preprocess')
os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, "text_preprocess.txt")
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

#%%
#Print the preprocessing of all text files
print("preprocessing done \n")
with open(output_file_path, 'r') as file:
    contents = file.read()
    print(contents)


# %%
#Proprocessing for CSV FILES
def preprocess_text(text):
    text = str(text).lower()  # Convert to lower case
    text = remove_special_characters(text)  # Remove special characters
    text = text.replace("NaN", "")
    text = text.replace("nan", "")# Rejoin words and remove "NaN"
    text = remove_stopwords_and_lemmatize(text)  # Process each word
    return text

def preprocess_csv(file_path, filename, output_dir):
    df = pd.read_csv(file_path)
    text_columns = df.select_dtypes(include=['object']).columns
    
    for column in text_columns:
        df[column] = df[column].apply(preprocess_text)
    new_file_name = filename.replace('.csv', '_processed.csv')
    
    # Save the preprocessed data to a new file in output_dir
    df.to_csv(os.path.join(output_dir, new_file_name), index=False)
    
    print(f"Preprocessed data saved to: {os.path.join(output_dir, new_file_name)}")
    print(df.head())

#%%
# Process all CSV file in the directory
directory = "../data/knowledge"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        preprocess_csv(file_path, filename, output_dir)

# %%
# Preprocess each csv file in the sample_q_and_a directory
directory = "../data/sample_q_and_a"
output_dir_qa = os.path.join(directory, 'preprocess')
os.makedirs(output_dir_qa, exist_ok=True)
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        preprocess_csv(file_path, filename, output_dir_qa)




# %%

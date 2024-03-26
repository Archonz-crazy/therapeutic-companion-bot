#%%
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from gensim.models.doc2vec import Doc2Vec
from gensim.models import FastText
import multiprocessing
from multiprocessing import Pool
import ast
import csv
import cProfile
import pstats
#%%
'''
directory = "../data/knowledge"
d2v_directory = os.path.join(directory, 'd2v')

os.makedirs(d2v_directory, exist_ok=True)

#%%
# Load data
# The path to your single text file
text_file_path = '../data/knowledge/text_preprocess.txt'

# Process the single text file
tagged_data = []

with open(text_file_path, 'r') as file:
    text = file.read()
    #Tokenize the text
    tokens = text.split()
    tagged_data.append(TaggedDocument(words=tokens, tags=["text_file"]))
#%%  
# Initialize and train the Doc2Vec model
model_d2v = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=multiprocessing.cpu_count(), epochs=100)
model_d2v.build_vocab(tagged_data)

#%%
# Train the Doc2Vec model
model_d2v.train(tagged_data, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)

#%%
# Save the model
model_d2v.save(os.path.join(d2v_directory, 'model_d2v.model'))
# %%
#view model_d2v.model file
from gensim.models.doc2vec import Doc2Vec
from concurrent.futures import ProcessPoolExecutor
model = Doc2Vec.load(os.path.join(d2v_directory, 'model_d2v.model'))
#%%
#view the document vector
document_tag = "text_file"
document_vector = model.dv[document_tag]
print(f"Vector for document '{document_tag}':\n{document_vector}")

# %%
# View a word vector
word = "hello"
word_vector = model.wv[word]
print(f"Vector for word '{word}':\n{word_vector}")
# %%
# List all document tags
all_document_tags = list(model.dv.index_to_key)
print("All document tags:", all_document_tags)

# %%
# Inspect the vocabulary
vocabulary = list(model.wv.key_to_index.keys())
print("Vocabulary:", vocabulary)

'''


# %%
# Process all text files in the knowledge directory
def process_text_file(directory):
    tagged_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                tokens = text.split() # Tokenize the text
                tagged_data.append(TaggedDocument(words=tokens, tags=["text_file"]))
    return tagged_data

#%%
'''
# Process all CSV files in the knowledge directory
def process_csv_files(directory):
    tagged_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
            
            # Identify all columns of type 'object' (string)
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if not text_columns:
                print(f"No suitable text columns found in {file_path}. Skipping file.")
                continue
            
            print(f"Using columns {text_columns} from {file_path}")
            for index, row in df.iterrows():
                # Assuming tokens are stored as a string representation of a list
                # and concatenating the text from all string columns
                tokens = []
                for col in text_columns:
                    column_data = row[col]
                    # Safely evaluate the string if it looks like a list, otherwise split normally
                    column_tokens = ast.literal_eval(column_data) if (isinstance(column_data, str) and column_data.startswith('[') and column_data.endswith(']')) else str(column_data).split()
                    tokens.extend(column_tokens)
                tagged_data.append(TaggedDocument(words=tokens, tags=[f"{os.path.basename(file_path)}_{index}"]))
    return tagged_data
'''
#%%
# Worker function to process a single CSV file
def process_single_csv(file_path):
    tagged_data = []
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')

        # Identify all columns of type 'object' (string)
        text_columns = [col for col in df.columns if df[col].dtype == 'object']

        if not text_columns:
            print(f"No suitable text columns found in {file_path}. Skipping file.")
            return tagged_data

        print(f"Using columns {text_columns} from {file_path}")
        for index, row in df.iterrows():
            tokens = []
            for col in text_columns:
                column_data = row[col]
                # Safely evaluate the string if it looks like a list, otherwise split normally
                column_tokens = ast.literal_eval(column_data) if (isinstance(column_data, str) and column_data.startswith('[') and column_data.endswith(']')) else str(column_data).split()
                tokens.extend(column_tokens)
            tagged_data.append(TaggedDocument(words=tokens, tags=[f"{os.path.basename(file_path)}_{index}"]))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return tagged_data

# Process all CSV files in the directory using multiprocessing
def process_csv_files(directory):
    files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".csv")]
    
    # Initialize a pool of worker processes
    with Pool(os.cpu_count()) as p:
        results = p.map(process_single_csv, files)

    # Flatten the list of tagged_data lists into a single list
    tagged_data = [item for sublist in results for item in sublist]
    return tagged_data

#%%
# Directory paths
directory = "../data/knowledge/preprocess"
d2v_directory = os.path.join(directory, 'd2v')
os.makedirs(d2v_directory, exist_ok=True)
# Initialize tagged data list
tagged_data = []

# Process the all text file that ends with .txt in the directory preprocess under knowledge folder
tagged_data.extend(process_text_file(directory))

# Process all CSV files in the preprocess directory of knowledge folder
tagged_data.extend(process_csv_files(directory))


#%%
# Train the Doc2Vec model for txt and csv files of knowledge folder using GPU
model_d2v_combined = Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.00025, min_count=2, dm=0, workers=multiprocessing.cpu_count(), epochs=10)
model_d2v_combined.build_vocab(tagged_data)
model_d2v_combined.train(tagged_data, total_examples=model_d2v_combined.corpus_count, epochs=model_d2v_combined.epochs)

# Save the model of knowledge folder
model_d2v_combined.save(os.path.join(d2v_directory, 'model_d2v_combined.model'))
# %%
# Directory paths for sample_q_and_a files
directory = "../data/sample_q_and_a/preprocess"
d2v_directory = os.path.join(directory, 'd2v')
os.makedirs(d2v_directory, exist_ok=True)
# Initialize tagged data list
tagged_data = []
# Process all CSV files in the directory preprocess under sample_qa folder
tagged_data.extend(process_csv_files(directory))

#%%
# Train the Doc2Vec model for sample_q_and_a folder
model_d2v_qa = Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=multiprocessing.cpu_count(), epochs=100)
model_d2v_qa.build_vocab(tagged_data)
model_d2v_qa.train(tagged_data, total_examples=model_d2v_qa.corpus_count, epochs=model_d2v_qa.epochs)
# Save the model of sample_q_and_a folder
model_d2v_qa.save(os.path.join(d2v_directory, 'model_d2v_qa.model'))


# %%
# view the model_d2v_qa.model file
for i in range(10):
    print(model_d2v_combined.docvecs[i])

#%%
#check the model_d2v_qa.model file for any word
for i in range(10):
    print(model_d2v_combined.wv.index_to_key[i])
    
# %%

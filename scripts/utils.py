#%%
# Imports
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#pytesseract.pytesseract.tesseract_cmd = r'/Users/mahi/anaconda3/envs/nlp/lib/python3.11/site-packages/pytesseract/pytesseract.py'

#%%
def convert_parquet_to_csv(directory):
    try:
        # Create a new directory for the CSV files
        csv_directory = os.path.join(directory, 'csv')
        os.makedirs(csv_directory, exist_ok=True)

        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(directory, filename))
                # Save the CSV files in the new directory
                df.to_csv(os.path.join(csv_directory, filename[:-8] + '.csv'), index=False)
        print("All parquet files have been converted to CSV.")
    except Exception as e:
        print(f"Error: {e}")
# %%
def split_column(column_name, keyword, file_path):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} does not exist in the DataFrame")

    # Split the column
    df[[f'Question', f'Response']] = df[column_name].str.split(keyword, expand=True, n=1)
    df.drop(column_name, axis=1, inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Column {column_name} successfully split into Question and Response")
#%%
def remove_string(path, column_name, string_to_remove):
    try:
        df = pd.read_csv(path)
        # Check if column_name is a valid column in df
        if column_name not in df.columns:
            print(f"Error: {column_name} is not a valid column in the DataFrame")
            return None

        # Check if the column contains None values
        if df[column_name].isnull().any():
            print(f"Error: {column_name} contains None values")
            return None

        df[column_name] = df[column_name].str.replace(string_to_remove, '')
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# %%
def concat_csv(file_path1, file_path2, output_file_path):
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        df = pd.concat([df1, df2])
        df.to_csv(output_file_path, index=False)
        print(f"CSV files successfully concatenated and saved to {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")

# %%
def drop_col(file_path, col_name):
    try:
        df = pd.read_csv(file_path)
        df.drop(col_name, axis=1, inplace=True)
        df.to_csv(file_path, index=False)
        print(f"Column {col_name} successfully dropped from {file_path}")
    except Exception as e:
        print(f"Error: {e}")
# %%
def remove_lines(path, col_name, string):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)

        # Remove rows where col_name does not contain string
        df = df[df[col_name].str.contains(string, na=False)]
        print("Lines with no responses are removed.")
        # Write the DataFrame back to the CSV file
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"Error: {e}")
# %%
def rename_col(path, col_replace, new_col):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)

        # using the rename() method 
        df.rename(columns = {col_replace : new_col}, inplace = True) 
        #df.columns = [col1, col2]
        print("Replace success.")
        
        # Write the DataFrame back to the CSV file
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"Error: {e}")

# %%
def concat_csv_list(file_list, path):
    try:
        csv_list = []
        for file in file_list:
            csv_list.append(pd.read_csv(file))
        print("csv list", csv_list)
        # 5. merges single pandas DFs into a single DF, index is refreshed 
        csv_merged = pd.concat(csv_list, ignore_index=True)
        
        # 6. Single DF is saved to the path in CSV format, without index column
        csv_merged.to_csv(path + 'all_types.csv', index=False)
        print("Merge completed")
    except Exception as e:
        print(f"Error: {e}")
        
# %%
def drop_columns(file_path, columns):
    try:
        df = pd.read_csv(file_path)
        df.drop(columns, axis=1, inplace=True)
        df.to_csv(file_path, index=False)
        print(f"Columns {columns} successfully dropped from {file_path}")
    except Exception as e:
        print(f"Error: {e}")
# %%
def json_to_csv(directory):
    try:
        # Create a new directory for the CSV files
        csv_directory = os.path.join(directory, 'csv')
        os.makedirs(csv_directory, exist_ok=True)

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                df = pd.read_json(os.path.join(directory, filename))
                # Save the CSV files in the new directory
                df.to_csv(os.path.join(csv_directory, filename[:-8] + '.csv'), index=False)
        print("All json files have been converted to CSV.")
    except Exception as e:
        print(f"Error: {e}")


# %%
from docx import Document
def doc_to_text(directory):
    try:
        # Create a new directory for the text files
        text_directory = os.path.join(directory, 'text')
        os.makedirs(text_directory, exist_ok=True)
        for filename in os.listdir(directory):
            if filename.endswith(".docx"):
                doc = Document(os.path.join(directory, filename))
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                with open(os.path.join(text_directory, filename[:-7] + '.txt'), 'w') as file:
                    file.write('\n'.join(full_text))
        print("All word files have been converted to text.")
    except Exception as e:
        print(f"Error: {e}")
# %%
def change_file_name(file_path, new_file_name):
    try:
        # Extract the directory and extension from the file path
        directory = os.path.dirname(file_path)
        extension = os.path.splitext(file_path)[1]

        # Create the new file path with the new file name and original extension
        new_file_path = os.path.join(directory, new_file_name + extension)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"File name changed to {new_file_name}")
    except Exception as e:
        print(f"Error: {e}")
#%%
# removing special characters using regex
def remove_special_characters(text):
    try:
        # Remove special characters, spaces, and empty lines using regex
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None
#%%
# removing stop words using nltk
def remove_stopwords(text):
    try:
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # Remove punctuations
        tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

        return tokens
    except Exception as e:
        print(f"Error: {e}")
        return None
# %%
# lemmatizing the text
def lemmatize_text(text):
    try:
        # Initialize the WordNetLemmatizer
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # Lemmatize the text
        text = [lemmatizer.lemmatize(word) for word in text]
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None
# %%
    
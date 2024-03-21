# All the imports
import os
import json
import pandas as pd
import re
import nltk
from docx import Document

def convert_to_csv(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                _, ext = os.path.splitext(file_path)
                if ext == '.json':
                    try:
                        df = pd.read_json(file_path, lines=True)  # Try to read file as line-delimited JSON
                    except ValueError:
                        df = pd.read_json(file_path)  # If error, try to read file as a regular JSON
                    output_dir = '/Users/mahi/Documents/therapeutic-companion-bot/data/processed/json'
                elif ext == '.parquet':
                    df = pd.read_parquet(file_path)
                    output_dir = '/Users/mahi/Documents/therapeutic-companion-bot/data/processed/parquet'
                else:
                    continue

                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Create CSV file path
                file_name = os.path.basename(file_path).rsplit('.', 1)[0] + '.csv'
                csv_file_path = os.path.join(output_dir, file_name)

                df.to_csv(csv_file_path, index=False)
                print(f"File has been converted to CSV: {csv_file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

class csv_processor:
    def concat_csv_list(file_list, path):
        try:
            csv_list = []
            for file in file_list:
                csv_list.append(pd.read_csv(file))
            print("csv list", csv_list)
            # 5. merges single pandas DFs into a single DF, index is refreshed 
            csv_merged = pd.concat(csv_list, ignore_index=True)
            csv_merged.to_csv(path + 'all_types.csv', index=False)
            print("Merge completed")
        except Exception as e:
            print(f"Error: {e}")

    def split_column(column_name, keyword, file_path):
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            raise ValueError(f"Column {column_name} does not exist in the DataFrame")

        # Split the column
        df[[f'Question', f'Response']] = df[column_name].str.split(keyword, expand=True, n=1)
        df.drop(column_name, axis=1, inplace=True)
        df.to_csv(file_path, index=False)
        print(f"Column {column_name} successfully split into Question and Response")

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
        
    def drop_col(file_path, col_name):
        try:
            df = pd.read_csv(file_path)
            df.drop(col_name, axis=1, inplace=True)
            df.to_csv(file_path, index=False)
            print(f"Column {col_name} successfully dropped from {file_path}")
        except Exception as e:
            print(f"Error: {e}")

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

class text_preprocessor:
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
        
    def remove_stopwords_and_lemmatize(text):
        try:
            # Tokenize the text
            tokens = nltk.word_tokenize(text)

            # Remove stopwords
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
            
            # Initialize the WordNetLemmatizer
            lemmatizer = nltk.stem.WordNetLemmatizer()
            # Lemmatize the text
            output = [lemmatizer.lemmatize(word) for word in tokens]
            return output
        except Exception as e:
            print(f"Error: {e}")
            return None

def main():
    PATH = os.getcwd()
    convert_to_csv(PATH+'/data/raw')
    

if __name__ == "__main__":

    main()



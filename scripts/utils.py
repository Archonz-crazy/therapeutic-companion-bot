#%%
# Imports
import pandas as pd
import os
import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
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
def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize a text holder
    text = ""
    
    # Iterate through each page
    for page in doc:
        # Extract text from the page and add it to the text holder
        text += page.get_text()
    
    # Close the document
    doc.close()
    
    return text
# %%
# PyMuPDF
def convert_pdf_to_txt(directory):
    try:
        # Create a new directory for the text files
        txt_directory = os.path.join(directory, 'txt')
        os.makedirs(txt_directory, exist_ok=True)

        img_directory = os.path.join(directory, 'images')
        os.makedirs(img_directory, exist_ok=True)

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                txt_path = os.path.join(txt_directory, filename[:-4] + '.txt')
                text = extract_text_from_pdf(pdf_path, img_directory)
                formatted_text = format_text(text)
                save_text_to_file(formatted_text, txt_path)
        print("All PDF files have been converted to TXT.")
    except Exception as e:
        print(f"Error: {e}")

def extract_text_from_pdf(pdf_path, img_directory):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize a text holder
    text = ""
    
    # Iterate through each page
    for page in doc:
        # Extract text from the page and add it to the text holder
        text += page.get_text()
    
        # Extract images from the page and perform OCR on them
        for img in page.get_images():
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]

    # Save the image to the images directory
            img_path = os.path.join(img_directory, img[-1])
            with open(img_path, 'wb') as f:
                f.write(image_data)

            img_text = extract_text_from_image(img_path)
            text += img_text
    # Close the document
    doc.close()
    
    return text

def extract_text_from_image(img_path):
    # Open the image file
    img = Image.open(img_path)
    
    # Perform OCR on the image using pytesseract
    img_text = pytesseract.image_to_string(img)
    
    return img_text

def format_text(text):
    # Format the text nicely, e.g., handle tables
    
    # TODO: Implement text formatting logic
    
    return text

def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

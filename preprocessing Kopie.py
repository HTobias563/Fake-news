import pandas as pd
import spacy
import os
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure the directory exists
output_dir = "./data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
file_path = "data/WELFake_Dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Define the enhanced cleaning function
def clean_text(text):
    text = text.strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Define a function to handle censored words
def handle_censored_words(text):
    return re.sub(r'\*+', '[CENSORED]', text)

# Define the preprocessing function for text
def preprocess_text_spacy(text):
    text = clean_text(text)
    text = handle_censored_words(text)
    doc = nlp(text.lower())
    filtered_sentence = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered_sentence)

# Define the preprocessing function for title
def preprocess_title_spacy(title):
    title = clean_text(title)
    title = handle_censored_words(title)
    doc = nlp(title.lower())
    filtered_sentence = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered_sentence)

# Apply the preprocessing functions to the 'text' and 'title' columns and overwrite them
df['text'] = df['text'].apply(preprocess_text_spacy)
df['title'] = df['title'].apply(preprocess_title_spacy)

# Handle the missing value in 'text' column
df.dropna(subset=['text'], inplace=True)
# or use: df['text'].fillna('missing', inplace=True)

# Save the preprocessed dataframe to a new CSV file without the index
df.to_csv(os.path.join(output_dir, "WELFake_Dataset_preprocessed_final.csv"), index=False)

# Print the first few rows to check the results
print(df.head(5))

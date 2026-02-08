import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import re
import argparse
import sys

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text_column(text_series):
    """
    Applies regex cleaning: lowercase, keep a-z only, normalize spaces.
    """
    return (text_series.str.lower()
            .str.replace(r'[^a-z\s]', ' ', regex=True)
            .str.replace(r' +', ' ', regex=True))

def process_tokens(text, stop_words):
    """
    Tokenizes and removes stop words/short words.
    """
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def process_file(input_path, output_path, stop_words, stemmer):
    print(f"Processing: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Warning: File {input_path} not found. Skipping.")
        return

    df = pd.read_csv(input_path)
    
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  - Dropped {initial_len - len(df)} duplicates")

    if 'sentiment' in df.columns:
        print("  - Mapping sentiment column...")
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    if 'review' in df.columns:
        print("  - Cleaning text regex...")
        df['review'] = clean_text_column(df['review'])
        
        print("  - Tokenizing and removing stopwords...")
        df['review'] = df['review'].apply(lambda x: process_tokens(x, stop_words))
        
        print("  - Stemming...")
        df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x])
    else:
        print("  - Warning: 'review' column missing.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    print(f"Saved processed data to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process NLP data for training and inference.")
    
    parser.add_argument('--input_dir', type=str, default='../../data/raw', help='Path to raw data')
    parser.add_argument('--output_dir', type=str, default='../../data/processed', help='Path to save processed data')
    
    args = parser.parse_args()

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    files_to_process = [
        ('train.csv', 'train.pkl'), 
        ('inference.csv', 'inference.pkl')
    ]

    for input_name, output_name in files_to_process:
        input_file = os.path.join(args.input_dir, input_name)
        output_file = os.path.join(args.output_dir, output_name)
        process_file(input_file, output_file, stop_words, stemmer)

if __name__ == "__main__":
    main()
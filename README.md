# Sentiment Analysis Pipeline
This project implements a complete end-to-end Machine Learning pipeline for sentiment analysis. It handles everything from downloading raw data and preprocessing text to training a Logistic Regression model and running inference on new datasets.

Each stage of the pipeline is containerized using Docker to ensure environment consistency and ease of deployment.

## Project Structure
The project is organized into modular components:
- src/data_loader/: Scripts to download and extract raw CSV datasets from remote URLs.

- src/data_processor/: Clean text (regex), removes stop words, and performs stemming using NLTK. (also performs vectorization - that's the reason why in data/processed files saved with pkl format)

- src/train/: Trains a Logistic Regression model using TF-IDF vectorization.

- src/inference/: Loads the trained model to predict sentiment on the inference dataset and outputs metrics.

- data/: Local directory for storing raw (CSV) and processed (PKL) data files.

- outputs/: Local directory for trained models and final predictions.

## Components & Usage
1. Data Loader
Downloads the training and inference zip files, extracts the CSVs, and flattens them into the target directory.
Input: Remote URLs. Output: train.csv, inference.csv in /data/raw.

2. Data Processor
Cleans the text by removing non-alphabetic characters, tokenizing, removing stop words, and applying Porter Stemming. (also performs vectorization - that's the reason why in data/processed files saved with pkl format)
Input: /data/raw/\*.csv.
Output: /data/processed/\*.pkl.

3. Model Training
Splits the processed training data (80/20 split), vectorizes text using TF-IDF, and fits a Logistic Regression model.
Metrics Generated: Accuracy, ROC-AUC, Classification Report, and Confusion Matrix.
Output: logreg_model.pkl, tfidf_vectorizer.pkl.

4. Inference
Loads the saved model and vectorizer to run predictions on the processed inference file.
Output: predictions.csv and metrics.txt.

## Getting Started with Docker
Each module contains its own Dockerfile. To run the pipeline, you can build and run each container sequentially, mounting local volumes to persist data between stages.

### Data loader
Build Command:
```bash
docker build -t data_loader ./src/data_loader
```

Run Command:
```bash
docker run -v $(pwd)/data:/data data_loader
```

### Data processor
Build Command:
```bash
docker build -t data_processor ./src/data_processor
```

Run Command:
```bash
docker run -v $(pwd)/data:/data data_processor
```


### Train
Build Command:
```bash
docker build -t train ./src/train
```

Run Command:
```bash
docker run -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs train
```


### Inference
Build Command:
```bash
docker build -t inference ./src/inference
```

Run Command:
```bash
docker run -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs inference
```

## Outputs and notebooks
Outputs have folder models and predictions

Models folder saves trained Logistic Regression 

Prediction holds predictions.csv with results of inference and metrics.txt with desctiption of model perfomance


Notebooks holds notebook with datascience part of task, and mlflow experiments

## Requirements
Python 3.9 (base image for containers).
Key Libraries: pandas, scikit-learn, nltk, joblib, requests.

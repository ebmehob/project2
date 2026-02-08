# Sentiment Analysis on Movie Reviews

## Project Overview
This project focuses on building a binary classification model to predict the sentiment (positive or negative) of movie reviews. The pipeline includes extensive Exploratory Data Analysis (EDA), text preprocessing (cleaning, tokenization, stemming/lemmatization), feature engineering using TF-IDF, and a comparison between a baseline Machine Learning model (Logistic Regression) and a custom Neural Network built with PyTorch.

## 1. Data Analysis & Preparation

### Dataset
* **Source:** The project utilizes a train dataset (`train.csv`) and an inference dataset (`inference.csv`).
* **Size:**  Train: 40,000 entries (reduced to 39,728 after deduplication).
    * Inference: 10,000 entries.
* **Target Variable:** `sentiment` (Binary: 1 for Positive, 0 for Negative).
* **Class Balance:** The dataset is perfectly balanced (50/50 split), making it ideal for evaluation without needing resampling techniques.

### Exploratory Data Analysis (EDA)
* **Data Quality:** Checked for missing values (None found).
* **Duplicates:** Identified and removed 272 duplicate rows in the training set and 13 in the inference set.
* **Text Length:** Analyzed review lengths; most are under 2,000 characters, with outliers up to 14,000 characters.

## 2. Feature Engineering

### Preprocessing Pipeline
1.  **Cleaning:** Converted text to lowercase, removed punctuation and numbers (keeping only `a-z` and spaces).
2.  **Tokenization:** Split text into individual tokens.
3.  **Stopword Removal:** Removed common English stopwords using NLTK.
4.  **Normalization Experiments:**
    * **Lemmatization (WordNet):** Accurate but computationally expensive due to POS tagging requirements.
    * **Stemming (Porter):** Faster but more aggressive (chopping word ends).
    * *Decision:* **Stemming** was chosen for the final pipeline as it provided a faster runtime and slightly better accuracy (approx. +0.5%) compared to Lemmatization for this specific dataset.

### Vectorization
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Used to convert text data into numerical vectors. This performed better than CountVectorizer.
* **Dimensionality Reduction:** For the Neural Network experiments, features were limited (`max_features=10000`) to optimize computation time.

## 3. Modeling & Results

The project compared a classic statistical approach against a deep learning approach.

### Baseline Model: Logistic Regression
* **Configuration:** Standard Logistic Regression.
* **Input:** Stemmed Text + TF-IDF Vectorization.
* **Performance:**
    * **Accuracy:** **88.79%**
    * *Note:* Outperformed the CountVectorizer variant (87.19%) and the Lemmatized variant (88.29%).

### Neural Network (PyTorch)
* **Architecture:** Custom Feed-Forward Neural Network (`TFIDF_NN`).
    * **Input Layer:** 10,000 features (TF-IDF).
    * **Hidden Layers:** 2 layers of 8 neurons each with ReLU activation.
    * **Regularization:** Dropout (0.5) to prevent overfitting.
    * **Output:** Single neuron (Logits) for binary classification.
* **Training:**
    * **Loss Function:** `BCEWithLogitsLoss`.
    * **Optimizer:** Adam (`lr=0.001`).
    * **Tracking:** Experiments tracked using **MLflow**.
* **Performance (Epoch 5):**
    * **Accuracy:** ~88.10%
    * **ROC AUC:** 0.9588
    * **F1 Score:** 0.8755

## 4. Conclusions & Key Findings

1.  **Simplicity Wins:** The baseline Logistic Regression model achieved the highest accuracy (~88.8%) with significantly faster training and inference times compared to the Neural Network.
2.  **Preprocessing Matters:** Stemming proved to be more effective than Lemmatization for this specific "Bag of Words" approach, likely due to the reduction of feature space dimensionality without losing critical sentiment signals.
3.  **Neural Network Limitation:** A simple Feed-Forward Network on TF-IDF data does not capture word order or context. 

### Future Improvements
To the 89% accuracy, the following approaches could be explored:
* **Word Embeddings:** Use Word2Vec, GloVe, or FastText to capture semantic meaning.
* **Deep Learning Architectures:** Implement **LSTMs** (Long Short-Term Memory).
* **Transformers:** Utilize pre-trained models like **BERT** or **RoBERTa**, which utilize attention mechanisms to understand context better than frequency-based methods.

## 5. Requirements (in requirements.txt - usage for notebook)
To run this notebook, the following libraries are required:
* `pandas`, `numpy`
* `nltk` (Stopwords, WordNet)
* `scikit-learn`
* `torch` (PyTorch)
* `mlflow`

## 6. Potential Business Applications and Value for Business

The sentiment analysis model developed in this project offers significant business value by automating the processing of vast amounts of unstructured text data, such as customer reviews and social media feedback. By deploying this solution, organizations can instantly gauge public opinion and brand reputation without manual effort, allowing for real-time "Voice of the Customer" analysis. This capability enables data-driven decision-making, such as prioritizing negative support tickets to reduce churn or identifying product weaknesses faster than competitors, ultimately transforming qualitative feedback into quantifiable, actionable business intelligence with approximately 89% accuracy.

# Sentiment Analysis Pipeline

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
Key Libraries: pandas, scikit-learn, nltk, joblib, requests. (all files have different requirements, that was hardcoded in Dockerfiles, since creating ones was decided to be not suitable)

import pandas as pd
import joblib
import os
import ast
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def identity_tokenizer(text):
    return text

def print_metrics(y_true, y_pred, y_prob=None, stage="Validation"):
    print(f"\n--- {stage} Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    
    if y_prob is not None:
        try:
            print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
        except ValueError:
            pass

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-----------------------\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/data/processed/train.pkl')
    parser.add_argument('--model_dir', type=str, default='/outputs/models')
    args = parser.parse_args()

    print(f"Loading data from {args.input_path}...")
    df = pd.read_pickle(args.input_path)
    
    X = df['review']
    y = df['sentiment']
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing...")
    tfidf = TfidfVectorizer(
        tokenizer=identity_tokenizer,  
        preprocessor=identity_tokenizer,
        token_pattern=None,
        max_features=5000
    )
    
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)

    print("Training Logistic Regression...")
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_val_vec)
    probs = clf.predict_proba(X_val_vec)[:, 1] # Probability for ROC-AUC
    print_metrics(y_val, preds, probs)

    os.makedirs(args.model_dir, exist_ok=True)
    
    model_path = os.path.join(args.model_dir, 'logreg_model.pkl')
    vectorizer_path = os.path.join(args.model_dir, 'tfidf_vectorizer.pkl')
    
    joblib.dump(clf, model_path)
    joblib.dump(tfidf, vectorizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    main()
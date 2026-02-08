import pandas as pd
import joblib
import os
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# for pickeled vectorizer
def identity_tokenizer(text):
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/data/processed/inference.pkl')
    parser.add_argument('--model_dir', type=str, default='/outputs/models')
    parser.add_argument('--output_path', type=str, default='/outputs/predictions/predictions.csv')
    parser.add_argument('--metrics_path', type=str, default='/outputs/predictions/metrics.txt')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, 'logreg_model.pkl')
    vec_path = os.path.join(args.model_dir, 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print(f"Error: Model artifacts not found at {args.model_dir}. Please run train.py first.")
        return

    print(f"Loading model and vectorizer from {args.model_dir}...")
    # joblib requires identity_tokenizer to be in scope if used during fit
    clf = joblib.load(model_path)
    tfidf = joblib.load(vec_path)

    print(f"Loading inference data from {args.input_path}...")
    df = pd.read_pickle(args.input_path)
    
    print("Running inference...")
    
    X_vec = tfidf.transform(df['review'])
    df['predicted_sentiment'] = clf.predict(X_vec)
    
    metrics_log = "--- Inference Metrics ---\n\n"

    if 'sentiment' in df.columns:
        print("\nGround truth found. Calculating metrics...")
        print("-----------------------------------------")
        y_true = df['sentiment']
        y_pred = df['predicted_sentiment']
        
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        conf_mat = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix:")
        print(conf_mat)

        metrics_log += f"Accuracy: {acc:.4f}\n\n"
        metrics_log += "Classification Report:\n"
        metrics_log += report + "\n"
        metrics_log += "Confusion Matrix:\n"
        metrics_log += str(conf_mat) + "\n"
    else:
        msg = "No ground truth ('sentiment') found. Skipping metrics.\n"
        print(f"\n{msg}")
        metrics_log += msg
        
        dist = df['predicted_sentiment'].value_counts().to_string()
        print("Prediction Distribution:")
        print(dist)
        metrics_log += "\nPrediction Distribution:\n" + dist

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"\nPredictions saved to {args.output_path}")

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w") as f:
        f.write(metrics_log)
    print(f"Metrics saved to {args.metrics_path}")

if __name__ == "__main__":
    main()
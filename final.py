import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSarcasmDetector:
    def __init__(self, output_dir='model_outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLTK components
        self._initialize_nltk()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=3,
            max_df=0.9,
            ngram_range=(1, 3)
        )
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

    def _initialize_nltk(self):
        """Download required NLTK resources."""
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {e}")
                raise

    def extract_linguistic_features(self, text):
        features = {}
        sentiment_scores = self.sia.polarity_scores(text)
        features.update(sentiment_scores)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        words = text.split()
        features['all_caps_words'] = sum(1 for word in words if word.isupper())
        return features

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s!?...]', '', text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def prepare_data(self, data_path):
        logger.info("Loading and preparing data...")
        df = pd.read_csv("Placement_projects\GEN-sar-notsarc.csv"
        )
        
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        linguistic_features = df['text'].apply(self.extract_linguistic_features)
        linguistic_df = pd.DataFrame.from_records(linguistic_features.values)
        
        X_tfidf = self.vectorizer.fit_transform(df['processed_text'])
        X_combined = np.hstack([X_tfidf.toarray(), linguistic_df.values])
        y = df['class'].map({'notsarc': 0, 'sarc': 1})
        
        return train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        logger.info("Starting model training...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def predict(self, text):
        processed_text = self.preprocess_text(text)
        linguistic_features = self.extract_linguistic_features(text)
        
        tfidf_features = self.vectorizer.transform([processed_text]).toarray()
        linguistic_array = np.array(list(linguistic_features.values())).reshape(1, -1)
        X_combined = np.hstack([tfidf_features, linguistic_array])
        
        prediction = self.model.predict(X_combined)
        probability = self.model.predict_proba(X_combined)
        
        return {
            'prediction': 'sarc' if prediction[0] == 1 else 'notsarc',
            'confidence': float(max(probability[0])),
            'explanation': self._generate_explanation(text)
        }

    def _generate_explanation(self, text):
        explanation = []
        sentiment = self.sia.polarity_scores(text)
        if abs(sentiment['compound']) > 0.5:
            explanation.append(f"Strong {'positive' if sentiment['compound'] > 0 else 'negative'} sentiment detected")
        
        features = self.extract_linguistic_features(text)
        if features['exclamation_count'] > 0:
            explanation.append("Contains exclamation marks")
        if features['all_caps_words'] > 0:
            explanation.append("Contains emphasized (ALL CAPS) words")
        return explanation
    
    def plot_evaluation_metrics(self, y_test, y_pred, y_pred_proba):
        """
        Create and save comprehensive evaluation plots.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Classification Metrics Plot
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        
        # Only keep the class-specific metrics (exclude averages and accuracy)
        metrics_df = metrics_df.iloc[:-3]  # Remove 'accuracy', 'macro avg', and 'weighted avg'
        
        # Drop the 'support' column
        if 'support' in metrics_df.columns:
            metrics_df = metrics_df.drop('support', axis=1)
        
        ax = metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Classification Metrics by Class')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.legend(title='Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
            
        plt.savefig(self.output_dir / f'classification_metrics_{timestamp}.png')
        plt.close()
        
        # 2. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        class_names = ['notsarc', 'sarc']  # Fixed class names based on mapping
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{timestamp}.png')
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_curve_{timestamp}.png')
        plt.close()
        
        # 4. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'precision_recall_curve_{timestamp}.png')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {self.output_dir}")
        
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance from the trained Random Forest model.
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.error("Model doesn't have feature importances attribute")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Take top 20 features for better visualization
        n_features = min(20, len(indices))
        top_indices = indices[:n_features]
        
        plt.figure(figsize=(10, 8))
        plt.title('Top Feature Importances')
        plt.barh(range(n_features), importances[top_indices], align='center')
        plt.yticks(range(n_features), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_importance_{timestamp}.png')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {self.output_dir}")

def main():
    try:
        logger.info("Starting sarcasm detection pipeline...")
        
        detector = EnhancedSarcasmDetector()
        data_path = "C:/Users/lezyo/Desktop/AD_Report/GEN-sarc-notsarc.csv"
        
        X_train, X_test, y_train, y_test = detector.prepare_data(data_path)
        detector.train(X_train, y_train)
        
        # Evaluate model and generate plots
        y_pred = detector.model.predict(X_test)
        y_pred_proba = detector.model.predict_proba(X_test)
        
        # Generate and save evaluation plots
        detector.plot_evaluation_metrics(y_test, y_pred, y_pred_proba)
        
        # Plot feature importance (with generic feature names since we don't have actual names)
        detector.plot_feature_importance()
        
        # Print classification report to console
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Test on sample texts
        test_texts = [
            "I absolutely love when my computer crashes!",
            "This is a great day for a picnic.",
            "Oh sure, because that's exactly what I needed right now."
        ]
        
        for text in test_texts:
            try:
                result = detector.predict(text)
                print(f"\nText: {text}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("Explanation:", ', '.join(result['explanation']))
            except Exception as e:
                logger.error(f"Prediction error for text '{text}': {e}")
                continue
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

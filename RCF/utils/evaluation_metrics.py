import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import os

class EvaluationMetrics:
    """Comprehensive evaluation metrics for binary classification"""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
    def add_prediction(self, true_label: str, predicted_label: str, probability: float = None):
        """Add a single prediction"""
        # Convert labels to binary (Positive=1, Negative=0)
        true_binary = 1 if true_label.lower() == 'positive' else 0
        pred_binary = 1 if predicted_label.lower() == 'positive' else 0
        
        self.true_labels.append(true_binary)
        self.predictions.append(pred_binary)
        
        if probability is not None:
            self.probabilities.append(probability)
        else:
            # If no probability provided, use 1.0 for correct predictions, 0.0 for incorrect
            self.probabilities.append(1.0 if true_binary == pred_binary else 0.0)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        if not self.true_labels:
            raise ValueError("No predictions added yet")
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # ROC AUC (if probabilities are available)
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.5  # Default for binary case
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'roc_auc': roc_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': len(y_true)
        }
        
        return self.metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.true_labels:
            raise ValueError("No predictions added yet")
        return confusion_matrix(self.true_labels, self.predictions)
    
    def print_report(self):
        """Print detailed classification report"""
        if not self.true_labels:
            raise ValueError("No predictions added yet")
        
        print("=" * 60)
        print("EVALUATION METRICS REPORT")
        print("=" * 60)
        
        # Calculate metrics if not already done
        if not self.metrics:
            self.calculate_metrics()
        
        # Print main metrics
        print(f"Total Samples: {self.metrics['total_samples']}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"MCC (Matthews Correlation Coefficient): {self.metrics['mcc']:.4f}")
        print(f"Specificity: {self.metrics['specificity']:.4f}")
        print(f"Sensitivity: {self.metrics['sensitivity']:.4f}")
        print(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}")
        print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        print("                Predicted")
        print("                Negative  Positive")
        print(f"Actual Negative    {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Actual Positive    {cm[1,0]:6d}    {cm[1,1]:6d}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            self.true_labels, 
            self.predictions, 
            target_names=['Negative', 'Positive'],
            zero_division=0
        ))
        
        print("=" * 60)
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        if not self.metrics:
            self.calculate_metrics()
        
        # Prepare data for saving
        results = {
            'metrics': self.metrics,
            'confusion_matrix': self.get_confusion_matrix().tolist(),
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'probabilities': self.probabilities
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix heatmap"""
        if not self.true_labels:
            raise ValueError("No predictions added yet")
        
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve"""
        if not self.true_labels or len(self.probabilities) == 0:
            print("Cannot plot ROC curve: no probability scores available")
            return
        
        y_true = np.array(self.true_labels)
        y_prob = np.array(self.probabilities)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve plot saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """Plot Precision-Recall curve"""
        if not self.true_labels or len(self.probabilities) == 0:
            print("Cannot plot Precision-Recall curve: no probability scores available")
            return
        
        y_true = np.array(self.true_labels)
        y_prob = np.array(self.probabilities)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve plot saved to: {save_path}")
        
        plt.show()

def evaluate_predictions_from_agents(agents: List, save_dir: str = "results/") -> EvaluationMetrics:
    """Evaluate predictions from a list of agents"""
    evaluator = EvaluationMetrics()
    
    for agent in agents:
        if agent.is_finished():
            true_label = agent.target
            predicted_label = agent.prediction
            
            # Extract probability from agent response if available
            probability = None
            try:
                # Try to extract confidence from agent response
                response = agent.scratchpad.split('Price Movement: ')[-1]
                if 'confidence' in response.lower():
                    # Extract confidence value if present
                    import re
                    confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', response.lower())
                    if confidence_match:
                        probability = float(confidence_match.group(1)) / 100.0
            except:
                pass
            
            evaluator.add_prediction(true_label, predicted_label, probability)
    
    return evaluator

def compare_models(model_results: Dict[str, EvaluationMetrics], save_dir: str = "results/"):
    """Compare multiple models and their performance"""
    comparison_data = []
    
    for model_name, evaluator in model_results.items():
        if evaluator.metrics:
            comparison_data.append({
                'Model': model_name,
                **evaluator.metrics
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Save comparison to CSV
        comparison_path = os.path.join(save_dir, "model_comparison.csv")
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(comparison_path, index=False)
        
        print("Model Comparison:")
        print(df.to_string(index=False))
        print(f"\nComparison saved to: {comparison_path}")
        
        return df
    
    return None 
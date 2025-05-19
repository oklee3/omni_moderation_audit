import csv
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

client = OpenAI()

def check_content(content):
    response = client.moderations.create(
        model = "omni-moderation-latest",
        input = content,
    )
    return response

def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1_score

def calculate_auc(y_true, y_scores):
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0

def save_metrics_and_scores(metrics, scores):
    # Save metrics to CSV
    with open('pilot_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])

        for name, (tp, tn, fp, fn) in metrics.items():
            acc, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn)
            auc = calculate_auc(scores[name]['true'], scores[name]['pred'])

            print(f"\nMetrics for '{name}':")
            print(f"  Accuracy: {acc:.2f}")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1 Score: {f1:.2f}")
            print(f"  AUC: {auc:.2f}")

            writer.writerow([name, acc, precision, recall, f1, auc])

def create_visualizations():
    # Read metrics
    df = pd.read_csv("pilot_output.csv")
    categories = df['Category']
    metrics = df.columns[1:]
    values = df.iloc[:, 1:].values

    # Create grouped bar plot
    x = np.arange(len(metrics))
    num_categories = len(categories)
    total_width = 0.8
    bar_width = total_width / num_categories

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (category, metric_values) in enumerate(zip(categories, values)):
        bar_positions = x - total_width / 2 + i * bar_width + bar_width / 2
        ax.bar(bar_positions, metric_values, bar_width, label=category)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(title="Category")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("visualizations/grouped_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.set_index('Category'), annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Performance Heatmap by Category')
    plt.ylabel('Category')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.savefig("visualizations/heatmap.png", dpi=300)
    plt.close()

    # Create Precision vs Recall plot
    plt.figure(figsize=(8, 6))
    for i, category in enumerate(categories):
        precision = values[i][list(metrics).index('Precision')]
        recall = values[i][list(metrics).index('Recall')]
        plt.scatter(recall, precision, label=category)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visualizations/precision_vs_recall.png", dpi=300)
    plt.close()

def run_audit(size=500):
    print(f"Starting audit with {size} samples...")
    
    # Load and sample data
    df = pd.read_csv("pilot_data/train.csv")
    df_random = df.sample(n=size, random_state=42)
    content_data = df_random.iloc[:, 1].tolist()
    label_data = df_random.iloc[:, 2:8].values.tolist()

    # Initialize metrics and scores
    metrics = {
        'flag': [0, 0, 0, 0],
        'obscene': [0, 0, 0, 0],
        'hate': [0, 0, 0, 0],
        'threat': [0, 0, 0, 0],
    }
    
    scores = {
        'flag': {'true': [], 'pred': []},
        'obscene': {'true': [], 'pred': []},
        'hate': {'true': [], 'pred': []},
        'threat': {'true': [], 'pred': []},
    }

    # Process each sample
    for i in range(size):
        if i % 50 == 0:
            print(f"Processing sample {i}/{size}...")
            
        result = check_content(content_data[i])
        labels = label_data[i]

        # Process each category
        # Flag category
        flag_true = 1 if any(labels) else 0
        scores['flag']['true'].append(flag_true)
        scores['flag']['pred'].append(float(result.results[0].flagged))
        
        if flag_true:
            metrics['flag'][0 if result.results[0].flagged else 3] += 1
        else:
            metrics['flag'][2 if result.results[0].flagged else 1] += 1

        # Obscene category
        obscene_true = 1 if labels[2] == 1 else 0
        obscene_pred = float(result.results[0].categories.sexual or 
                           result.results[0].categories.illicit or 
                           result.results[0].categories.violence or 
                           result.results[0].categories.self_harm)
        scores['obscene']['true'].append(obscene_true)
        scores['obscene']['pred'].append(obscene_pred)
        
        if obscene_true:
            metrics['obscene'][0 if obscene_pred else 3] += 1
        else:
            metrics['obscene'][2 if obscene_pred else 1] += 1

        # Hate category
        hate_true = 1 if labels[5] == 1 else 0
        hate_pred = float(result.results[0].categories.hate)
        scores['hate']['true'].append(hate_true)
        scores['hate']['pred'].append(hate_pred)
        
        if hate_true:
            metrics['hate'][0 if hate_pred else 3] += 1
        else:
            metrics['hate'][2 if hate_pred else 1] += 1

        # Threat category
        threat_true = 1 if labels[5] == 1 else 0
        threat_pred = float(getattr(result.results[0].categories, "hate/threatening", 0) or 
                          getattr(result.results[0].categories, "harassment/threatening", 0))
        scores['threat']['true'].append(threat_true)
        scores['threat']['pred'].append(threat_pred)
        
        if threat_true:
            metrics['threat'][0 if threat_pred else 3] += 1
        else:
            metrics['threat'][2 if threat_pred else 1] += 1

    print("\nSaving metrics and scores...")
    save_metrics_and_scores(metrics, scores)
    
    print("\nCreating visualizations...")
    create_visualizations()
    
    print("\nAudit complete! Check the visualizations directory for plots.")

if __name__ == "__main__":
    run_audit() 
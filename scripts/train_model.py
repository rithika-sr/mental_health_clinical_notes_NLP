"""
Mental Health Crisis Detection - Model Training
Step 3: Train transformer model for severity classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("STEP 3: MODEL TRAINING")
print("="*80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load preprocessed data
print("\n1. Loading preprocessed data...")
df = pd.read_csv('../data/processed/mental_health_preprocessed.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Display class distribution
print("\n   Class distribution:")
class_dist = df['severity_label'].value_counts().sort_index()
severity_mapping = {0: "No Risk", 1: "Mild Risk", 2: "Moderate Risk", 3: "High Risk"}
for label, count in class_dist.items():
    pct = (count / len(df)) * 100
    print(f"   {severity_mapping[label]:15}: {count:6,} ({pct:5.2f}%)")

# Train-validation-test split
print("\n2. Splitting data into train, validation, and test sets...")
# First split: 80% train+val, 20% test
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df['severity_label']
)

# Second split: 75% train (of remaining 80%), 25% val (of remaining 80%)
# This gives us 60% train, 20% val, 20% test overall
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=42,
    stratify=train_val_df['severity_label']
)

print(f"   Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

# Verify stratification
print("\n   Verifying stratification:")
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['severity_label'].value_counts(normalize=True).sort_index() * 100
    print(f"   {split_name:10}: ", end="")
    for label in range(4):
        print(f"Class {label}: {dist[label]:5.2f}%  ", end="")
    print()

# Initialize tokenizer
print("\n3. Initializing tokenizer...")
model_name = "distilbert-base-uncased"
print(f"   Model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Convert to Hugging Face datasets
print("\n4. Converting to Hugging Face dataset format...")
train_dataset = HFDataset.from_pandas(train_df[['text', 'severity_label']])
val_dataset = HFDataset.from_pandas(val_df[['text', 'severity_label']])
test_dataset = HFDataset.from_pandas(test_df[['text', 'severity_label']])

# Tokenize datasets
print("   Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Rename label column
train_dataset = train_dataset.rename_column('severity_label', 'labels')
val_dataset = val_dataset.rename_column('severity_label', 'labels')
test_dataset = test_dataset.rename_column('severity_label', 'labels')

# Set format for PyTorch
train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

print("   Tokenization complete")

# Load model
print("\n5. Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    problem_type="single_label_classification"
)
model.to(device)
print(f"   Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Define metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }

# Create models directory
os.makedirs('../models', exist_ok=True)
os.makedirs('../models/logs', exist_ok=True)

# Training arguments
print("\n6. Setting up training configuration...")
training_args = TrainingArguments(
    output_dir='../models/mental_health_classifier',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../models/logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1_weighted',
    greater_is_better=True,
    save_total_limit=2,
    report_to="none"
)

print("   Training configuration:")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Warmup steps: {training_args.warmup_steps}")

# Initialize Trainer
print("\n7. Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train model
print("\n8. Starting training...")
print("   This may take 10-30 minutes depending on your hardware...")
print("   " + "-"*76)

train_result = trainer.train()

print("\n   Training complete!")
print(f"   Final training loss: {train_result.training_loss:.4f}")

# Evaluate on validation set
print("\n9. Evaluating on validation set...")
val_results = trainer.evaluate()
print("   Validation Results:")
print(f"   - Accuracy: {val_results['eval_accuracy']:.4f}")
print(f"   - F1 (Weighted): {val_results['eval_f1_weighted']:.4f}")
print(f"   - F1 (Macro): {val_results['eval_f1_macro']:.4f}")

# Evaluate on test set
print("\n10. Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print("   Test Results:")
print(f"   - Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"   - F1 (Weighted): {test_results['eval_f1_weighted']:.4f}")
print(f"   - F1 (Macro): {test_results['eval_f1_macro']:.4f}")

# Generate predictions
print("\n11. Generating predictions for confusion matrix...")
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_df['severity_label'].values

# Confusion Matrix
print("\n12. Creating confusion matrix...")
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Risk', 'Mild', 'Moderate', 'High'],
            yticklabels=['No Risk', 'Mild', 'Moderate', 'High'])
plt.title('Confusion Matrix - Mental Health Crisis Detection', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../data/processed/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   Saved: ../data/processed/confusion_matrix.png")
plt.close()

# Classification Report
print("\n13. Classification Report:")
print("-" * 80)
report = classification_report(
    true_labels, 
    pred_labels,
    target_names=['No Risk', 'Mild', 'Moderate', 'High'],
    digits=4
)
print(report)

# Save classification report
with open('../data/processed/classification_report.txt', 'w') as f:
    f.write("Mental Health Crisis Detection - Classification Report\n")
    f.write("="*80 + "\n\n")
    f.write(report)
print("   Saved: ../data/processed/classification_report.txt")

# Save model
print("\n14. Saving trained model...")
model.save_pretrained('../models/mental_health_classifier_final')
tokenizer.save_pretrained('../models/mental_health_classifier_final')
print("   Model saved to: ../models/mental_health_classifier_final")

# Save test set for later inference
print("\n15. Saving test set for inference...")
test_df.to_csv('../data/processed/test_set.csv', index=False)
print("   Test set saved to: ../data/processed/test_set.csv")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Final Test F1 (Weighted): {test_results['eval_f1_weighted']:.4f}")
print("\nNext steps:")
print("1. Review confusion matrix: ../data/processed/confusion_matrix.png")
print("2. Review classification report: ../data/processed/classification_report.txt")
print("3. Test model with sample predictions")
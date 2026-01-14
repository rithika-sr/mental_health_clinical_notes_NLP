"""
Mental Health Crisis Detection - Model Training (Improved)
Step 3: Train BioClinicalBERT for better clinical text understanding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
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
print("STEP 3: MODEL TRAINING (IMPROVED WITH BIOCLINICALBERT)")
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
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df['severity_label']
)

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

# Compute class weights for imbalanced data
print("\n3. Computing class weights for imbalanced data...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['severity_label']),
    y=train_df['severity_label']
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("   Class weights:")
for label, weight in class_weights_dict.items():
    print(f"   {severity_mapping[label]:15}: {weight:.4f}")

# Initialize tokenizer
print("\n4. Initializing BioClinicalBERT tokenizer...")
model_name = "emilyalsentzer/Bio_ClinicalBERT"
print(f"   Model: {model_name}")
print("   This model is specifically trained on clinical notes!")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

# Convert to Hugging Face datasets
print("\n5. Converting to Hugging Face dataset format...")
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
print("\n6. Loading BioClinicalBERT pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    problem_type="single_label_classification"
)
model.to(device)
print(f"   Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Custom Trainer with class weights - FIXED FOR MPS/CPU
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Create class weights tensor on the same device as the model
        weight_tensor = torch.tensor(
            list(class_weights_dict.values()), 
            dtype=torch.float32,
            device=logits.device  # Use the same device as logits
        )
        
        # Apply class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Define metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, predictions, average=None)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_no_risk': f1_per_class[0],
        'f1_mild': f1_per_class[1],
        'f1_moderate': f1_per_class[2],
        'f1_high': f1_per_class[3]
    }

# Create models directory
os.makedirs('../models', exist_ok=True)
os.makedirs('../models/logs', exist_ok=True)

# Training arguments
print("\n7. Setting up training configuration...")
training_args = TrainingArguments(
    output_dir='../models/mental_health_bioclinical',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../models/logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    save_total_limit=2,
    report_to="none"
)

print("   Training configuration:")
print(f"   - Model: BioClinicalBERT (clinical text specialist)")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Max sequence length: 256 tokens")
print(f"   - Class weights: Enabled (balances minority classes)")

# Initialize Trainer
print("\n8. Initializing Weighted Trainer...")
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train model
print("\n9. Starting training...")
print("   This may take 20-40 minutes depending on your hardware...")
print("   BioClinicalBERT will provide better clinical understanding!")
print("   " + "-"*76)

train_result = trainer.train()

print("\n   Training complete!")
print(f"   Final training loss: {train_result.training_loss:.4f}")

# Evaluate on validation set
print("\n10. Evaluating on validation set...")
val_results = trainer.evaluate()
print("   Validation Results:")
print(f"   - Accuracy: {val_results['eval_accuracy']:.4f}")
print(f"   - F1 (Weighted): {val_results['eval_f1_weighted']:.4f}")
print(f"   - F1 (Macro): {val_results['eval_f1_macro']:.4f}")
print(f"   - F1 No Risk: {val_results['eval_f1_no_risk']:.4f}")
print(f"   - F1 Mild: {val_results['eval_f1_mild']:.4f}")
print(f"   - F1 Moderate: {val_results['eval_f1_moderate']:.4f}")
print(f"   - F1 High Risk: {val_results['eval_f1_high']:.4f}")

# Evaluate on test set
print("\n11. Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print("   Test Results:")
print(f"   - Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"   - F1 (Weighted): {test_results['eval_f1_weighted']:.4f}")
print(f"   - F1 (Macro): {test_results['eval_f1_macro']:.4f}")
print(f"   - F1 No Risk: {test_results['eval_f1_no_risk']:.4f}")
print(f"   - F1 Mild: {test_results['eval_f1_mild']:.4f}")
print(f"   - F1 Moderate: {test_results['eval_f1_moderate']:.4f}")
print(f"   - F1 High Risk: {test_results['eval_f1_high']:.4f}")

# Generate predictions
print("\n12. Generating predictions for confusion matrix...")
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_df['severity_label'].values

# Confusion Matrix
print("\n13. Creating confusion matrix...")
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Risk', 'Mild', 'Moderate', 'High'],
            yticklabels=['No Risk', 'Mild', 'Moderate', 'High'])
plt.title('Confusion Matrix - BioClinicalBERT', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../data/processed/confusion_matrix_bioclinical.png', dpi=300, bbox_inches='tight')
print("   Saved: ../data/processed/confusion_matrix_bioclinical.png")
plt.close()

# Classification Report
print("\n14. Classification Report:")
print("-" * 80)
report = classification_report(
    true_labels, 
    pred_labels,
    target_names=['No Risk', 'Mild', 'Moderate', 'High'],
    digits=4
)
print(report)

# Save classification report
with open('../data/processed/classification_report_bioclinical.txt', 'w') as f:
    f.write("Mental Health Crisis Detection - BioClinicalBERT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Max Sequence Length: 256 tokens\n")
    f.write(f"Class Weights: Enabled\n\n")
    f.write(report)
    f.write("\n\nClass Weights Used:\n")
    for label, weight in class_weights_dict.items():
        f.write(f"{severity_mapping[label]:15}: {weight:.4f}\n")
print("   Saved: ../data/processed/classification_report_bioclinical.txt")

# Save model
print("\n15. Saving trained BioClinicalBERT model...")
model.save_pretrained('../models/mental_health_bioclinical_final')
tokenizer.save_pretrained('../models/mental_health_bioclinical_final')
print("   Model saved to: ../models/mental_health_bioclinical_final")

# Save test set
print("\n16. Saving test set for inference...")
test_df.to_csv('../data/processed/test_set.csv', index=False)
print("   Test set saved to: ../data/processed/test_set.csv")

# Compare with DistilBERT if previous results exist
print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nBioClinicalBERT Results:")
print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"  Test F1 (Macro): {test_results['eval_f1_macro']:.4f}")
print(f"  F1 per class:")
print(f"    No Risk:   {test_results['eval_f1_no_risk']:.4f}")
print(f"    Mild:      {test_results['eval_f1_mild']:.4f}")
print(f"    Moderate:  {test_results['eval_f1_moderate']:.4f}")
print(f"    High Risk: {test_results['eval_f1_high']:.4f}")
print("\nKey Improvements:")
print("  - Clinical text understanding (trained on medical notes)")
print("  - Better negation handling")
print("  - Balanced class weights")

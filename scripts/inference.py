"""
Mental Health Crisis Detection - Inference
Step 4: Test the trained model with sample predictions
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Severity mapping
severity_mapping = {
    0: "No Risk",
    1: "Mild Risk", 
    2: "Moderate Risk",
    3: "High Risk"
}

# Color coding for terminal output
colors = {
    0: '\033[92m',  # Green
    1: '\033[93m',  # Yellow
    2: '\033[91m',  # Orange/Red
    3: '\033[95m',  # Magenta
    'RESET': '\033[0m'
}

print("="*80)
print("MENTAL HEALTH CRISIS DETECTION - INFERENCE")
print("="*80)

# Load model and tokenizer
print("\nLoading trained model...")
model_path = '../models/mental_health_classifier_final'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Model loaded successfully!")

def predict_severity(text, show_probabilities=True):
    """
    Predict severity level for given text
    
    Args:
        text: Clinical note or mental health text
        show_probabilities: Whether to show probability distribution
    
    Returns:
        Dictionary with prediction and probabilities
    """
    
    # Tokenize
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get all probabilities
    all_probs = {
        severity_mapping[i]: probabilities[0][i].item() 
        for i in range(4)
    }
    
    result = {
        'text': text,
        'predicted_severity': severity_mapping[predicted_class],
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs
    }
    
    return result

def display_prediction(result):
    """Display prediction in a formatted way"""
    
    pred_class = result['predicted_class']
    color = colors[pred_class]
    reset = colors['RESET']
    
    print("\n" + "-"*80)
    print(f"Text: {result['text'][:200]}...")
    print(f"\n{color}Predicted Severity: {result['predicted_severity']}{reset}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    
    print("\nProbability Distribution:")
    for severity, prob in result['probabilities'].items():
        bar_length = int(prob * 50)
        bar = '█' * bar_length
        print(f"  {severity:15}: {bar} {prob*100:5.2f}%")
    print("-"*80)

# Test with sample clinical notes
print("\n" + "="*80)
print("TESTING WITH SAMPLE CLINICAL NOTES")
print("="*80)

# Sample texts representing each severity level
sample_texts = [
    {
        'text': "Patient reports feeling well today. No complaints. Good mood and affect. Plans to continue current medication regimen.",
        'expected': 'No Risk'
    },
    {
        'text': "Patient mentions feeling stressed about work deadlines. Reports occasional anxiety but managing well with coping strategies.",
        'expected': 'Mild Risk'
    },
    {
        'text': "Patient expresses feelings of hopelessness about future. Reports severe depression symptoms. Difficulty functioning at work.",
        'expected': 'Moderate Risk'
    },
    {
        'text': "Patient reports active suicidal ideation with plan. States no reason to live. Requires immediate intervention.",
        'expected': 'High Risk'
    }
]

print("\nRunning predictions on sample texts...")

for i, sample in enumerate(sample_texts, 1):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i} (Expected: {sample['expected']})")
    print('='*80)
    
    result = predict_severity(sample['text'])
    display_prediction(result)
    
    # Check if prediction matches expectation
    if result['predicted_severity'] == sample['expected']:
        print(f"\n✓ Prediction matches expected severity!")
    else:
        print(f"\n✗ Prediction differs from expected (Got: {result['predicted_severity']}, Expected: {sample['expected']})")

# Interactive mode
print("\n" + "="*80)
print("INTERACTIVE MODE")
print("="*80)
print("\nYou can now enter your own clinical notes for prediction.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter clinical note (or 'quit' to exit): ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nExiting inference mode. Goodbye!")
        break
    
    if len(user_input.strip()) < 10:
        print("Please enter a longer text (at least 10 characters).")
        continue
    
    result = predict_severity(user_input)
    display_prediction(result)
    print()

print("\n" + "="*80)
print("INFERENCE COMPLETE")
print("="*80)
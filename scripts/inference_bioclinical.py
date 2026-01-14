"""
Mental Health Crisis Detection - BioClinicalBERT Inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

severity_mapping = {0: "No Risk", 1: "Mild Risk", 2: "Moderate Risk", 3: "High Risk"}
colors = {0: '\033[92m', 1: '\033[93m', 2: '\033[91m', 3: '\033[95m', 'RESET': '\033[0m'}

print("="*80)
print("MENTAL HEALTH CRISIS DETECTION - BioClinicalBERT")
print("="*80)

print("\nLoading BioClinicalBERT model...")
model_path = '../models/mental_health_bioclinical_final'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Model loaded!")

def predict(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    return {
        'severity': severity_mapping[pred_class],
        'class': pred_class,
        'confidence': probs[0][pred_class].item(),
        'all_probs': {severity_mapping[i]: probs[0][i].item() for i in range(4)}
    }

def display(result):
    color = colors[result['class']]
    print(f"\n{color}→ {result['severity']}{colors['RESET']} ({result['confidence']*100:.1f}% confident)")
    for sev, prob in result['all_probs'].items():
        bar = '█' * int(prob * 40)
        print(f"  {sev:15}: {bar} {prob*100:5.1f}%")

# Critical test cases
print("\n" + "="*80)
print("TESTING CRITICAL CASES (The ones that failed before)")
print("="*80)

tests = [
    ("Patient is not suicidal", "Should be: No Risk"),
    ("Patient denies suicidal thoughts and is doing well", "Should be: No Risk"),  
    ("Patient reports active suicidal ideation with plan", "Should be: High Risk"),
    ("Patient feels hopeless and can't cope anymore", "Should be: Moderate Risk")
]

for text, expected in tests:
    print(f"\nTest: '{text}'")
    print(f"Expected: {expected}")
    result = predict(text)
    display(result)

# Interactive
print("\n" + "="*80)
print("INTERACTIVE MODE - Type 'quit' to exit")
print("="*80)

while True:
    text = input("\nEnter clinical note: ")
    if text.lower() in ['quit', 'q', 'exit']:
        break
    if len(text.strip()) < 10:
        print("Enter longer text (10+ characters)")
        continue
    result = predict(text)
    display(result)

print("\nDone!")
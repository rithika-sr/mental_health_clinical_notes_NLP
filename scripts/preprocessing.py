"""
Mental Health Crisis Detection - Data Preprocessing
Step 2: Create severity labels and preprocess text data
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
sns.set_style("whitegrid")

print("="*80)
print("STEP 2: DATA PREPROCESSING AND SEVERITY LABEL CREATION")
print("="*80)

# Load datasets with correct file paths (files are in data/raw/)
print("\n1. Loading datasets...")
df_reddit = pd.read_csv('../data/raw/Mental_Health_Reddit.csv')
df_twitter = pd.read_csv('../data/raw/Mental-Health-Twitter.csv')

print(f"   Reddit dataset: {df_reddit.shape}")
print(f"   Twitter dataset: {df_twitter.shape}")

# Prepare Twitter dataset
print("\n2. Preparing Twitter dataset...")
df_twitter_clean = df_twitter[['post_text', 'label']].copy()
df_twitter_clean.rename(columns={'post_text': 'text'}, inplace=True)

# Add source tracking
df_reddit['source'] = 'reddit'
df_twitter_clean['source'] = 'twitter'

# Combine datasets
print("\n3. Combining datasets...")
df_combined = pd.concat([df_reddit, df_twitter_clean], ignore_index=True)
print(f"   Combined dataset shape: {df_combined.shape}")

# Define severity keywords based on clinical mental health indicators
print("\n4. Defining severity classification keywords...")

severity_keywords = {
    'high_risk': [
        r'\bsuicid',
        r'\bkill myself\b',
        r'\bend my life\b',
        r'\bwant to die\b',
        r'\bending it all\b',
        r'\bno reason to live\b',
        r'\bself.harm\b',
        r'\bharm myself\b',
        r'\bcut myself\b',
        r'\boverdose\b'
    ],
    'moderate_risk': [
        r'\bsevere depression\b',
        r'\bhopeless',
        r'\bworthless',
        r'\bcan.t cope\b',
        r'\bcan.t go on\b',
        r'\bgiving up\b',
        r'\bno point\b',
        r'\bnumb\b',
        r'\bempty inside\b',
        r'\bnothing matters\b',
        r'\bpanic attack',
        r'\bbreakdown\b'
    ],
    'mild_risk': [
        r'\bdepressed\b',
        r'\bdepress',
        r'\banxious\b',
        r'\banxiety\b',
        r'\bworried\b',
        r'\bstressed\b',
        r'\bsad\b',
        r'\blonely\b',
        r'\bunhappy\b',
        r'\bdown\b',
        r'\bmental health\b',
        r'\btherapy\b',
        r'\bmedication\b'
    ]
}

def classify_severity(text, original_label):
    """
    Classify text into 4 severity levels:
    0 = No risk (original label was 0)
    1 = Mild risk (mentions depression/anxiety without severe symptoms)
    2 = Moderate risk (severe symptoms, hopelessness)
    3 = High risk (suicidal ideation or self-harm)
    """
    
    # If original label is 0 (no mental health issue), keep it as 0
    if original_label == 0:
        return 0
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Check for high risk indicators first (highest priority)
    for pattern in severity_keywords['high_risk']:
        if re.search(pattern, text_lower):
            return 3
    
    # Check for moderate risk indicators
    for pattern in severity_keywords['moderate_risk']:
        if re.search(pattern, text_lower):
            return 2
    
    # Check for mild risk indicators
    for pattern in severity_keywords['mild_risk']:
        if re.search(pattern, text_lower):
            return 1
    
    # If label was 1 but no specific keywords found, default to mild
    return 1

# Apply severity classification
print("\n5. Applying severity labels (this may take a minute)...")
df_combined['severity_label'] = df_combined.apply(
    lambda row: classify_severity(row['text'], row['label']), 
    axis=1
)

# Analyze severity distribution
print("\n6. Analyzing severity label distribution...")
severity_counts = df_combined['severity_label'].value_counts().sort_index()
severity_percentages = df_combined['severity_label'].value_counts(normalize=True).sort_index() * 100

severity_mapping = {
    0: "No Risk",
    1: "Mild Risk", 
    2: "Moderate Risk",
    3: "High Risk"
}

print("\n   Severity Label Distribution:")
print("   " + "-"*60)
for severity, count in severity_counts.items():
    percentage = severity_percentages[severity]
    label_name = severity_mapping[severity]
    print(f"   Label {severity} ({label_name:15}): {count:6,} samples ({percentage:5.2f}%)")

# Visualize severity distribution
print("\n7. Creating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

# Bar plot
severity_counts.plot(kind='bar', ax=axes[0], color=colors, alpha=0.8)
axes[0].set_title('Severity Label Distribution (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Severity Level', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No Risk\n(0)', 'Mild\n(1)', 'Moderate\n(2)', 'High\n(3)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(severity_counts):
    axes[0].text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
axes[1].pie(severity_counts, labels=[severity_mapping[i] for i in severity_counts.index],
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Severity Label Distribution (Percentages)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../data/severity_distribution.png', dpi=300, bbox_inches='tight')
print("   Saved: ../data/severity_distribution.png")
plt.close()

# Display sample texts by severity
print("\n8. Sample texts by severity level:")
print("   " + "="*76)

for severity in range(4):
    print(f"\n   SEVERITY {severity}: {severity_mapping[severity]}")
    print("   " + "-"*76)
    
    samples = df_combined[df_combined['severity_label'] == severity]['text'].head(3)
    
    for idx, text in enumerate(samples, 1):
        display_text = text[:200] + "..." if len(text) > 200 else text
        print(f"   Sample {idx}: {display_text}\n")

# Text preprocessing function
print("\n9. Defining text preprocessing function...")

def preprocess_text(text):
    """
    Clean and preprocess text data
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions (@username)
    4. Remove hashtag symbols but keep text
    5. Remove special characters and numbers
    6. Remove extra whitespace
    """
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbols but keep the text
    text = re.sub(r'#', '', text)
    
    # Remove special characters and numbers but keep spaces and basic punctuation
    text = re.sub(r'[^a-z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing
print("\n10. Preprocessing text data...")
df_combined['text_clean'] = df_combined['text'].apply(preprocess_text)

# Show preprocessing examples
print("\n    Preprocessing examples:")
print("    " + "-"*76)
for idx in range(2):
    print(f"\n    Original: {df_combined.iloc[idx]['text'][:150]}...")
    print(f"    Cleaned:  {df_combined.iloc[idx]['text_clean'][:150]}...")

# Filter out very short texts
print("\n11. Filtering out very short texts...")
df_combined['text_length_clean'] = df_combined['text_clean'].str.len()
rows_before = len(df_combined)
df_combined = df_combined[df_combined['text_length_clean'] >= 10].copy()
rows_after = len(df_combined)

print(f"    Rows before filtering: {rows_before:,}")
print(f"    Rows after filtering:  {rows_after:,}")
print(f"    Rows removed:          {rows_before - rows_after:,}")

df_combined.reset_index(drop=True, inplace=True)

# Final dataset summary
print("\n12. Final dataset summary:")
print("    " + "="*76)
print(f"    Total samples: {len(df_combined):,}")

print("\n    Severity distribution:")
for severity, count in df_combined['severity_label'].value_counts().sort_index().items():
    pct = (count / len(df_combined)) * 100
    print(f"    {severity_mapping[severity]:15}: {count:6,} ({pct:5.2f}%)")

print("\n    Text length statistics (cleaned):")
stats = df_combined['text_length_clean'].describe()
for stat_name, stat_value in stats.items():
    print(f"    {stat_name:8}: {stat_value:.2f}")

# Save preprocessed data
print("\n13. Saving preprocessed dataset...")
df_final = df_combined[['text_clean', 'severity_label', 'source']].copy()
df_final.rename(columns={'text_clean': 'text'}, inplace=True)

output_path = '../data/mental_health_preprocessed.csv'
df_final.to_csv(output_path, index=False)

print(f"    Saved to: {output_path}")
print(f"    Shape: {df_final.shape}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("\nNext step: Train-test split and model training")
print("Run: python 03_train_model.py")
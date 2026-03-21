import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ── STEP 1: URL Feature Functions ────────────────────────

def get_url_length(url):
    return len(url)

def get_dot_count(url):
    return url.count('.')

def get_slash_count(url):
    return url.count('/')

def has_ip_address(url):
    pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(pattern, url) else 0

def has_at_symbol(url):
    return 1 if '@' in url else 0

def has_https(url):
    return 1 if url.startswith('https') else 0

def has_port(url):
    return 1 if re.search(r':\d+', url.split('//')[-1]) else 0

def get_domain_length(url):
    try:
        domain = url.split('//')[-1].split('/')[0]
        return len(domain)
    except:
        return 0

def get_subdomain_count(url):
    try:
        domain = url.split('//')[-1].split('/')[0]
        parts = domain.split('.')
        return max(0, len(parts) - 2)
    except:
        return 0

def has_suspicious_words(url):
    suspicious = ['login', 'verify', 'secure', 'account',
                  'update', 'confirm', 'banking', 'paypal',
                  'password', 'signin', 'authenticate']
    url_lower = url.lower()
    return 1 if any(word in url_lower for word in suspicious) else 0

def has_hyphen(url):
    try:
        domain = url.split('//')[-1].split('/')[0]
        return 1 if '-' in domain else 0
    except:
        return 0

def get_digit_count(url):
    return sum(c.isdigit() for c in url)

def get_special_char_count(url):
    return sum(c in '!#$%^&*~`' for c in url)

def has_double_slash(url):
    return 1 if '//' in url[7:] else 0

def get_path_length(url):
    try:
        path = url.split('//')[-1].split('/', 1)
        return len(path[1]) if len(path) > 1 else 0
    except:
        return 0

def has_shortener(url):
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl',
                  't.co', 'ow.ly', 'short.io']
    return 1 if any(s in url.lower() for s in shorteners) else 0

def extract_features(url):
    return [
        get_url_length(url),
        get_dot_count(url),
        get_slash_count(url),
        has_ip_address(url),
        has_at_symbol(url),
        has_https(url),
        has_port(url),
        get_domain_length(url),
        get_subdomain_count(url),
        has_suspicious_words(url),
        has_hyphen(url),
        get_digit_count(url),
        get_special_char_count(url),
        has_double_slash(url),
        get_path_length(url),
        has_shortener(url),
    ]

FEATURE_NAMES = [
    'url_length', 'dot_count', 'slash_count',
    'has_ip', 'has_at', 'has_https', 'has_port',
    'domain_length', 'subdomain_count',
    'suspicious_words', 'has_hyphen',
    'digit_count', 'special_chars',
    'double_slash', 'path_length', 'has_shortener'
]

# ── STEP 2: Load datasets and combine ────────────────────
print("Loading datasets...")

# Dataset 1: URLs from our email dataset
df_emails = pd.read_csv('clean_emails.csv')
df_emails['urls'] = df_emails['urls'].apply(ast.literal_eval)
df_emails_with_urls = df_emails[df_emails['urls'].apply(len) > 0].copy()
df_emails_with_urls['url'] = df_emails_with_urls['urls'].apply(lambda x: x[0])
df_email_urls = df_emails_with_urls[['url', 'label']].copy()
df_email_urls.columns = ['url', 'label']
print(f"Email dataset URLs: {len(df_email_urls)}")

# Dataset 2: Malicious URLs dataset (much larger)
df_malicious = pd.read_csv('malicious_phish.csv')

# Keep only phishing and benign
df_malicious = df_malicious[df_malicious['type'].isin(['phishing', 'benign'])]

# Convert labels to numbers
df_malicious['label'] = df_malicious['type'].map({
    'benign': 0,
    'phishing': 1
})
df_malicious = df_malicious[['url', 'label']]
print(f"Malicious URLs dataset: {len(df_malicious)}")

# Balance: 50,000 benign + all phishing
df_benign = df_malicious[df_malicious['label'] == 0].sample(50000, random_state=42)
df_phishing = df_malicious[df_malicious['label'] == 1]
df_malicious_balanced = pd.concat([df_benign, df_phishing])

# Combine both datasets
df_combined = pd.concat([df_email_urls, df_malicious_balanced], ignore_index=True)
df_combined = df_combined.dropna(subset=['url'])
df_combined = df_combined[df_combined['url'].str.strip() != '']
print(f"Combined dataset: {len(df_combined)} URLs")
print(f"Label distribution:\n{df_combined['label'].value_counts()}")

# ── STEP 3: Extract features ──────────────────────────────
print("Extracting URL features...")
features = df_combined['url'].apply(extract_features)
X = pd.DataFrame(features.tolist(), columns=FEATURE_NAMES)
y = df_combined['label'].reset_index(drop=True)
print(f"Dataset shape: {X.shape}")

# ── STEP 4: Train/test split ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── STEP 5: Train model ───────────────────────────────────
print("Training URL feature model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
print("Training complete!")

# ── STEP 6: Evaluate ──────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Phishing']))

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}  (Target: >= 0.70)")

# ── STEP 7: Confusion matrix ──────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('URL Feature Model - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('url_confusion_matrix.png')
print("Confusion matrix saved!")

# ── STEP 8: Feature importance ────────────────────────────
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(FEATURE_NAMES)),
        importances[indices], color='steelblue')
plt.xticks(range(len(FEATURE_NAMES)),
           [FEATURE_NAMES[i] for i in indices],
           rotation=45, ha='right')
plt.title('URL Feature Importance')
plt.tight_layout()
plt.savefig('url_feature_importance.png')
print("Feature importance chart saved!")

# ── STEP 9: Save model ────────────────────────────────────
joblib.dump(model, 'url_model.pkl')
print("URL model saved as url_model.pkl")
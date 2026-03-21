import pandas as pd
import numpy as np
import joblib
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ── Load all saved models ─────────────────────────────────
print("Loading models...")
nlp_model = joblib.load('nlp_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
url_model = joblib.load('url_model.pkl')
print("All models loaded!")

# ── URL Feature Extraction ────────────────────────────────
# Copied here directly so we don't retrain url_model.py

FEATURE_NAMES = [
    'url_length', 'dot_count', 'slash_count',
    'has_ip', 'has_at', 'has_https', 'has_port',
    'domain_length', 'subdomain_count',
    'suspicious_words', 'has_hyphen',
    'digit_count', 'special_chars',
    'double_slash', 'path_length', 'has_shortener'
]

def extract_features(url):
    def get_url_length(url): return len(url)
    def get_dot_count(url): return url.count('.')
    def get_slash_count(url): return url.count('/')
    def has_ip_address(url): return 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    def has_at_symbol(url): return 1 if '@' in url else 0
    def has_https(url): return 1 if url.startswith('https') else 0
    def has_port(url): return 1 if re.search(r':\d+', url.split('//')[-1]) else 0
    def get_domain_length(url):
        try: return len(url.split('//')[-1].split('/')[0])
        except: return 0
    def get_subdomain_count(url):
        try:
            parts = url.split('//')[-1].split('/')[0].split('.')
            return max(0, len(parts) - 2)
        except: return 0
    def has_suspicious_words(url):
        words = ['login','verify','secure','account','update',
                 'confirm','banking','paypal','password','signin','authenticate']
        return 1 if any(w in url.lower() for w in words) else 0
    def has_hyphen(url):
        try: return 1 if '-' in url.split('//')[-1].split('/')[0] else 0
        except: return 0
    def get_digit_count(url): return sum(c.isdigit() for c in url)
    def get_special_char_count(url): return sum(c in '!#$%^&*~`' for c in url)
    def has_double_slash(url): return 1 if '//' in url[7:] else 0
    def get_path_length(url):
        try:
            path = url.split('//')[-1].split('/', 1)
            return len(path[1]) if len(path) > 1 else 0
        except: return 0
    def has_shortener(url):
        return 1 if any(s in url.lower() for s in ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','short.io']) else 0

    return [
        get_url_length(url), get_dot_count(url), get_slash_count(url),
        has_ip_address(url), has_at_symbol(url), has_https(url),
        has_port(url), get_domain_length(url), get_subdomain_count(url),
        has_suspicious_words(url), has_hyphen(url), get_digit_count(url),
        get_special_char_count(url), has_double_slash(url),
        get_path_length(url), has_shortener(url),
    ]

# ── Score functions ───────────────────────────────────────

def get_nlp_score(text):
    text_tfidf = vectorizer.transform([text])
    return float(nlp_model.predict_proba(text_tfidf)[0][1])

def get_url_score(url):
    features = extract_features(url)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    return float(url_model.predict_proba(features_df)[0][1])

# ── Fusion weights ────────────────────────────────────────
NLP_WEIGHT = 0.6
URL_WEIGHT = 0.4

def combine_scores(nlp_score, url_score=None):
    if url_score is None:
        final_score = nlp_score
        method = "NLP only"
    else:
        final_score = (NLP_WEIGHT * nlp_score) + (URL_WEIGHT * url_score)
        method = f"Weighted (NLP={NLP_WEIGHT}, URL={URL_WEIGHT})"

    if final_score >= 0.5:
        verdict = "PHISHING"
    elif final_score >= 0.3:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"

    return final_score, verdict, method

# ── Load dataset ──────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv('clean_emails.csv')
df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']
df['urls'] = df['urls'].apply(ast.literal_eval)
print(f"Total emails: {len(df)}")

# ── Get NLP scores ────────────────────────────────────────
print("Getting NLP scores...")
nlp_scores = df['clean_text'].apply(get_nlp_score).values

# ── Get URL scores ────────────────────────────────────────
print("Getting URL scores...")
def safe_url_score(urls):
    if len(urls) > 0:
        return get_url_score(urls[0])
    return None

url_scores = df['urls'].apply(safe_url_score).values

# ── Combine scores ────────────────────────────────────────
print("Combining scores...")
final_scores = []
for nlp_s, url_s in zip(nlp_scores, url_scores):
    score, verdict, method = combine_scores(nlp_s, url_s)
    final_scores.append(score)

final_scores = np.array(final_scores)
y_true = df['label'].values
y_pred_nlp_only = (nlp_scores >= 0.5).astype(int)

# Evaluate combined only on emails that HAVE urls
# This is the fair comparison — both models have data
has_url = np.array([len(u) > 0 for u in df['urls']])
y_true_with_url = y_true[has_url]
nlp_scores_with_url = nlp_scores[has_url]
url_scores_with_url = np.array([
    get_url_score(u[0]) for u in df['urls'][has_url]
])

# Combined prediction on emails with URLs
combined_with_url = (
    NLP_WEIGHT * nlp_scores_with_url +
    URL_WEIGHT * url_scores_with_url
)
y_pred_combined = (combined_with_url >= 0.5).astype(int)
y_pred_nlp_subset = (nlp_scores_with_url >= 0.5).astype(int)

# ── Results ───────────────────────────────────────────────
print("\n--- NLP Only Results ---")
print(classification_report(y_true, y_pred_nlp_only,
      target_names=['Legitimate', 'Phishing']))

print("\n--- Combined System Results (emails with URLs only) ---")
print(classification_report(y_true_with_url, y_pred_combined,
      target_names=['Legitimate', 'Phishing']))

f1_nlp = f1_score(y_true_with_url, y_pred_nlp_subset)
f1_combined = f1_score(y_true_with_url, y_pred_combined)
print(f"NLP only F1:  {f1_nlp:.4f}")
print(f"Combined F1:  {f1_combined:.4f}")
diff = f1_combined - f1_nlp
print(f"Difference:   {diff:+.4f}")

# ── Confusion matrix ──────────────────────────────────────
cm = confusion_matrix(y_true_with_url, y_pred_combined)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Combined System - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('combined_confusion_matrix.png')
print("\nCombined confusion matrix saved!")

# ── Comparison chart ──────────────────────────────────────
# URL only score on emails that have URLs
emails_with_urls = df[df['urls'].apply(len) > 0]
url_scores_filtered = np.array([get_url_score(u[0]) for u in emails_with_urls['urls']])
y_true_url = emails_with_urls['label'].values
y_pred_url = (url_scores_filtered >= 0.5).astype(int)
f1_url = f1_score(y_true_url, y_pred_url)

models = ['NLP Only', 'URL Only', 'Combined']
scores = [f1_nlp, f1_url, f1_combined]
colors = ['steelblue', 'green', 'coral']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=colors)
plt.axhline(y=0.88, color='red', linestyle='--',
            label='Excellent target (0.88)')
plt.axhline(y=0.80, color='orange', linestyle='--',
            label='Proficient target (0.80)')
plt.ylim(0, 1.0)
plt.title('F1 Score: Individual vs Combined Models')
plt.ylabel('F1 Score')
plt.legend()
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Model comparison chart saved!")
print("\nStep 6 COMPLETE!")
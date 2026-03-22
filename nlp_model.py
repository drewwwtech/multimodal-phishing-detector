import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# ── Load clean dataset ────────────────────────────────────
df = pd.read_csv('clean_emails.csv')
print(f"Total emails: {len(df)}")
print(df['label'].value_counts())

# ── Split into training and testing sets ─────────────────
# 80% training, 20% testing
df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']
print(f"After final cleaning: {len(df)} emails remaining")

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ── Convert text to numbers using TF-IDF ─────────────────
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# ── Train the model ───────────────────────────────────────
print("Training model... please wait")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("Training complete!")

# ── Evaluate the model ────────────────────────────────────
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Phishing']))

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}  (Target: >= 0.80)")

# ── Plot confusion matrix ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('NLP Model — Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('nlp_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved as nlp_confusion_matrix.png")

# ── Save the model ────────────────────────────────────────
joblib.dump(model, 'nlp_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model saved as nlp_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")

# ── Show top phishing words ───────────────────────────────
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

top_phishing_idx = np.argsort(coef)[-20:][::-1]

words = [feature_names[i] for i in top_phishing_idx]
scores = [coef[i] for i in top_phishing_idx]

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], scores[::-1], color='steelblue')
plt.title('Top 20 Words Most Associated With Phishing')
plt.xlabel('TF-IDF Weight')
plt.tight_layout()
plt.savefig('nlp_feature_importance.png')
print("Feature importance chart saved!")
plt.show()

plt.savefig('nlp_confusion_matrix.png')
print("Confusion matrix saved as nlp_confusion_matrix.png")
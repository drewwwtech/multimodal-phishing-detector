import pandas as pd
import re
from bs4 import BeautifulSoup

# ── STEP 1: Load the dataset ──────────────────────────────
df = pd.read_csv('Phishing_Email.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ── STEP 2: Remove empty emails ──────────────────────────
df = df.dropna(subset=['Email Text'])
print(f"After removing empty emails: {len(df)}")

# ── STEP 3: Convert labels to numbers ────────────────────
df['label'] = df['Email Type'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})
print(df['Email Type'].value_counts())

# ── STEP 4: Clean each email ─────────────────────────────
def clean_email(text):
    urls = re.findall(r'https?://\S+', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[-_=]{3,}', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text, urls

print("Cleaning all emails... please wait")
results = df['Email Text'].apply(clean_email)
df['clean_text'] = results.apply(lambda x: x[0])
df['urls'] = results.apply(lambda x: x[1])

# ── STEP 5: Save clean dataset ───────────────────────────
df[['clean_text', 'urls', 'label']].to_csv('clean_emails.csv', index=False)
print(f"Done! Saved {len(df)} clean emails to clean_emails.csv")
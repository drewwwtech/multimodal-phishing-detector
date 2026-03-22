import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import ast
import time
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Phishing Detector",
    page_icon="🎣",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import ast

    nlp_model = joblib.load('nlp_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Retrain URL model if pkl doesn't exist
    if os.path.exists('url_model.pkl'):
        url_model = joblib.load('url_model.pkl')
    else:
        st.info("Setting up URL model for first time... (~30 seconds)")

        # Load email URLs
        df_emails = pd.read_csv('clean_emails.csv')
        df_emails['urls'] = df_emails['urls'].apply(ast.literal_eval)
        df_emails_with_urls = df_emails[
            df_emails['urls'].apply(len) > 0].copy()
        df_emails_with_urls['url'] = df_emails_with_urls['urls'].apply(
            lambda x: x[0])
        df_email_urls = df_emails_with_urls[['url', 'label']].copy()

        # Load malicious URLs dataset
        df_malicious = pd.read_csv('malicious_phish.csv')
        df_malicious = df_malicious[
            df_malicious['type'].isin(['phishing', 'benign'])]
        df_malicious['label'] = df_malicious['type'].map(
            {'benign': 0, 'phishing': 1})
        df_malicious = df_malicious[['url', 'label']]

        # Use smaller sample for cloud (still accurate)
        df_benign = df_malicious[
            df_malicious['label'] == 0].sample(20000, random_state=42)
        df_phishing = df_malicious[
            df_malicious['label'] == 1].sample(20000, random_state=42)
        df_combined = pd.concat(
            [df_email_urls, df_benign, df_phishing], ignore_index=True)
        df_combined = df_combined.dropna(subset=['url'])
        df_combined = df_combined[df_combined['url'].str.strip() != '']

        # Extract features and train
        features = df_combined['url'].apply(extract_features)
        X = pd.DataFrame(features.tolist(), columns=FEATURE_NAMES)
        y = df_combined['label'].reset_index(drop=True)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        url_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced'
        )
        url_model.fit(X_train, y_train)
        joblib.dump(url_model, 'url_model.pkl')
        st.success("URL model ready!")

    return nlp_model, vectorizer, url_model

# ── URL Feature Extraction ────────────────────────────────
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
        return 1 if any(s in url.lower() for s in
                       ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','short.io']) else 0

    return [
        get_url_length(url), get_dot_count(url), get_slash_count(url),
        has_ip_address(url), has_at_symbol(url), has_https(url),
        has_port(url), get_domain_length(url), get_subdomain_count(url),
        has_suspicious_words(url), has_hyphen(url), get_digit_count(url),
        get_special_char_count(url), has_double_slash(url),
        get_path_length(url), has_shortener(url),
    ]

# ── Helper functions ──────────────────────────────────────

def extract_urls(text):
    pattern = r'https?://[^\s<>"\'()]+'
    return re.findall(pattern, text)

def clean_text(text):
    urls = re.findall(r'https?://\S+', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[-_=]{3,}', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower(), urls

def get_nlp_score(text):
    cleaned, _ = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    return float(nlp_model.predict_proba(tfidf)[0][1])

def get_url_score(url):
    features = extract_features(url)
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    return float(url_model.predict_proba(df)[0][1])

def take_screenshot(url, save_path='temp_screenshot.png'):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--disable-gpu")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(10)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(save_path)
        driver.quit()
        return save_path
    except Exception as e:
        return None

def combine_scores(nlp_score, url_score=None):
    NLP_WEIGHT = 0.6
    URL_WEIGHT = 0.4
    if url_score is None:
        final_score = nlp_score
        method = "NLP only (no URL found)"
    else:
        final_score = (NLP_WEIGHT * nlp_score) + (URL_WEIGHT * url_score)
        method = f"Weighted average (NLP x{NLP_WEIGHT} + URL x{URL_WEIGHT})"
    return final_score, method

# ── App Layout ────────────────────────────────────────────

st.title("Multi-Modal Phishing Detector")
st.markdown(
    "Paste an email below to analyze it for phishing using "
    "**NLP text analysis** and **URL feature detection**."
)
st.divider()

# ── Input ─────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Paste Email Here")
    email_input = st.text_area(
        label="email",
        height=200,
        placeholder="Paste the full email text here including any links...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Settings")
    enable_screenshot = st.checkbox(
        "Capture website screenshot",
        value=True,
        help="Automatically visits and screenshots any link found in the email"
    )
    st.info(
        "The detector automatically balances between "
        "analyzing the **email text** (60%) and the "
        "**linked website** (40%). Adjust the slider below to change how much weight each component has in the final score."
    )
    nlp_weight = st.slider("NLP Weight", 0.0, 1.0, 0.6, 0.1)
    url_weight = round(1.0 - nlp_weight, 1)
    st.write(f"URL Weight: {url_weight}")

# ── Analyze button ────────────────────────────────────────
analyze = st.button("Analyze Email", type="primary",
                    use_container_width=True)

if analyze and email_input.strip():
    st.divider()
    st.subheader("Analysis Results")

    progress = st.progress(0, text="Starting analysis...")

    # NLP analysis
    progress.progress(25, text="Running NLP text analysis...")
    nlp_score = get_nlp_score(email_input)

    # Extract URLs
    urls_found = extract_urls(email_input)
    first_url = urls_found[0] if urls_found else None

    # URL analysis
    url_score = None
    if first_url:
        progress.progress(50, text=f"Analyzing URL features...")
        url_score = get_url_score(first_url)

    # Screenshot
    screenshot_path = None
    if first_url and enable_screenshot:
        progress.progress(70, text=f"Screenshotting website...")
        screenshot_path = take_screenshot(first_url)

    # Combine
    progress.progress(90, text="Combining scores...")
    final_score, method = combine_scores(nlp_score, url_score)
    progress.progress(100, text="Done!")
    progress.empty()

    # ── Verdict ───────────────────────────────────────────
    if final_score >= 0.5:
        st.error(f"VERDICT: PHISHING — Do NOT interact with this email!")
    elif final_score >= 0.3:
        st.warning(f"VERDICT: SUSPICIOUS — Proceed with caution")
    else:
        st.success(f"VERDICT: LEGITIMATE — Email appears safe")

    # ── Score breakdown ───────────────────────────────────
    col_nlp, col_url, col_final = st.columns(3)

    with col_nlp:
        st.metric("NLP Text Score", f"{nlp_score:.0%}")

    with col_url:
        st.metric("URL Score",
                  f"{url_score:.0%}" if url_score is not None else "N/A")

    with col_final:
        st.metric("Final Combined Score", f"{final_score:.0%}")

    st.info(f"Fusion method: {method}")

    # ── URLs found ────────────────────────────────────────
    if urls_found:
        st.subheader("URLs Found in Email")
        for url in urls_found:
            st.code(url)

    # ── Screenshot ────────────────────────────────────────
    if screenshot_path and os.path.exists(screenshot_path):
        st.subheader("Website Screenshot")
        img = Image.open(screenshot_path)
        st.image(img, caption="Screenshot of linked website",
                 use_container_width=True)

    # ── Score breakdown table ─────────────────────────────
    with st.expander("How was this score calculated?"):
        st.markdown(f"""
        | Component | Score | Weight | Contribution |
        |-----------|-------|--------|--------------|
        | NLP Text Analysis | {nlp_score:.0%} | {nlp_weight} | {nlp_score * nlp_weight:.0%} |
        | URL Feature Analysis | {f"{url_score:.0%}" if url_score else "N/A"} | {url_weight} | {(url_score or 0) * url_weight:.0%} |
        | **Final Score** | **{final_score:.0%}** | — | — |

        **Method:** {method}
        """)

elif analyze:
    st.warning("Please paste email content before clicking Analyze.")

st.divider()
st.caption("IAS1 Cybersecurity Project — Group 4 | For educational purposes only")
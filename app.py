import pandas as pd
import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords kalau belum
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Regex cleaning
    text = re.sub(r"http\S+|www.\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)             # Mention
    text = re.sub(r"#\w+", "", text)             # Hashtag
    text = re.sub(r"\brt\b", "", text)           # Retweet
    text = re.sub(r"[^a-z\s]", "", text)         # Punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()     # Extra spaces

    # 3. Stopword removal
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

def get_top_contributing_words(text, top_n=5):
    kalimat_bersih = clean_text(text)
    tokens = kalimat_bersih.split()
    vektor = vectorizer.transform([kalimat_bersih])

    # Ambil fitur (kata) dan bobot
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    # Simpan kontribusi tiap kata
    contributions = {}
    for token in tokens:
        if token in feature_names:
            idx = vectorizer.vocabulary_.get(token)
            if idx is not None:
                contributions[token] = weights[idx]

    # Urutkan dari kontribusi terbesar (positif = indikasi mental issue)
    top_words = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return top_words


# Fungsi prediksi
def prediksi_kalimat_LR(text):
    kalimat_bersih = clean_text(text)
    vektor = vectorizer.transform([kalimat_bersih])
    pred = model.predict(vektor)[0]
    prob = model.predict_proba(vektor)[0]
    confidence = max(prob) * 100

    hasil_label = "🔴Terindikasi Mental Health Issue" if pred == 1 else "🟢Tidak Terindikasi"
    return hasil_label, confidence

# Streamlit App
st.set_page_config(page_title="Mental Health Issue Detection", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Issue Detection")
st.markdown("-----------------------------")
st.markdown("Tell what you are feeling honestly (bad/good) like you are confiding in your friend!!")
st.markdown("-- Write in 10-20 words in English!! --")

input_user = st.text_area("📝 Your Feelings :")

if st.button("🔍 Prediction"):
    if input_user.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        hasil, confidence = prediksi_kalimat_LR(input_user)
        st.success(f"{hasil} (Probabilitas: {confidence:.2f}%)")

        top_kata = get_top_contributing_words(input_user)
        if top_kata:
            st.markdown("**Kata-kata paling berpengaruh dalam prediksi:**")
            for kata, skor in top_kata:
                arah = "🔺" if skor > 0 else "🔻"
                st.markdown(f"- {kata} {arah} (bobot: {skor:.4f})")

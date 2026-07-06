
import streamlit as st
import pandas as pd
import numpy as np
import re

from collections import Counter
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="Wondr by BNI Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#fff7ed 0%,
#f8fafc 50%,
#ecfeff 100%);
}

.main-title{
font-size:38px;
font-weight:800;
color:#0f172a;
}

.sub-title{
font-size:16px;
color:#475569;
}

.metric-card{
background:white;
padding:20px;
border-radius:18px;
box-shadow:0px 4px 20px rgba(0,0,0,0.08);
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("wondr_bni_reviews.csv")
df["clean_review"] = df["review"]
df.columns = df.columns.str.strip()

df["review"] = df["review"].fillna("").astype(str)

# =====================================
# LABEL SENTIMEN
# =====================================

def sentiment_label(rating):

    if rating >= 4:
        return "positive"

    elif rating == 3:
        return "neutral"

    else:
        return "negative"

if "polarity" not in df.columns:
    df["polarity"] = df["rating"].apply(sentiment_label)

# =====================================
# TF-IDF
# =====================================

tfidf = TfidfVectorizer(max_features=3000)

df["clean_review"] = df["clean_review"].fillna("").astype(str)

X = tfidf.fit_transform(df["clean_review"])
X = tfidf.fit_transform(df["clean_review"])

y = df["polarity"]

# =====================================
# SPLIT
# =====================================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# NAIVE BAYES
# =====================================

model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

df["predicted_sentiment"] = model.predict(X)

df["review_length"] = df["review"].apply(len)

with st.sidebar:

    st.title("📊 Wondr Analytics")

    menu = st.radio(
        "Menu",
        [
            "Dashboard",
            "Dataset",
            "Analisis Sentimen",
            "WordCloud",
            "Prediksi Sentimen"
        ]
    )

# =====================================
# DASHBOARD
# =====================================

if menu == "Dashboard":

    st.markdown(
        "<div class='main-title'>📊 Wondr by BNI Analytics</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Analisis Sentimen Ulasan Pengguna Menggunakan Naive Bayes</div>",
        unsafe_allow_html=True
    )

    st.write("")

    total_review = len(df)

    avg_rating = round(df["rating"].mean(),2)

    dominant_sentiment = (
        df["polarity"]
        .value_counts()
        .idxmax()
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total Review",f"{total_review:,}")

    c2.metric("Rata-rata Rating",avg_rating)

    c3.metric(
        "Akurasi Model",
        f"{accuracy*100:.2f}%"
    )

    c4.metric(
        "Sentimen Dominan",
        dominant_sentiment
    )

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        sentiment_count = (
            df["polarity"]
            .value_counts()
            .reset_index()
        )

        sentiment_count.columns = [
            "Sentiment",
            "Jumlah"
        ]

        fig = px.pie(
            sentiment_count,
            names="Sentiment",
            values="Jumlah",
            hole=0.4,
            title="Distribusi Sentimen"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        rating_count = (
            df["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_count.columns = [
            "Rating",
            "Jumlah"
        ]

        fig2 = px.bar(
            rating_count,
            x="Rating",
            y="Jumlah",
            title="Distribusi Rating"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =====================================
# DATASET
# =====================================

elif menu == "Dataset":

    st.subheader("📄 Dataset Ulasan")

    st.dataframe(
        df[
            [
                "review",
                "rating",
                "polarity"
            ]
        ],
        use_container_width=True
    )

# =====================================
# ANALISIS SENTIMEN
# =====================================

elif menu == "Analisis Sentimen":

    st.subheader("📊 Analisis Sentimen")

    sentiment_count = (
        df["polarity"]
        .value_counts()
        .reset_index()
    )

    sentiment_count.columns = [
        "Sentiment",
        "Jumlah"
    ]

    fig = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Jumlah",
        color="Sentiment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# WORDCLOUD
# =====================================

elif menu == "WordCloud":

    st.subheader("☁️ WordCloud")

    text = " ".join(
        df["clean_review"]
        .astype(str)
        .tolist()
    )

    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color="white"
    ).generate(text)

    fig,ax = plt.subplots(
        figsize=(12,5)
    )

    ax.imshow(wordcloud)

    ax.axis("off")

    st.pyplot(fig)

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    freq = Counter(words)

    top_words = pd.DataFrame(
        freq.most_common(20),
        columns=[
            "Kata",
            "Frekuensi"
        ]
    )

    st.dataframe(
        top_words,
        use_container_width=True
    )

# =====================================
# PREDIKSI
# =====================================

elif menu == "Prediksi Sentimen":

    st.subheader(
        "🤖 Prediksi Sentimen"
    )

    review = st.text_area(
        "Masukkan Ulasan"
    )

    if st.button("Prediksi"):

        review = review.lower()

        vector = tfidf.transform(
            [review]
        )

        prediksi = model.predict(
            vector
        )[0]

        if prediksi == "positive":

            st.success(
                "😊 Sentimen Positif"
            )

        elif prediksi == "neutral":

            st.warning(
                "😐 Sentimen Netral"
            )

        else:

            st.error(
                "😞 Sentimen Negatif"
            )

import rootutils

ROOT = rootutils.autosetup()

import numpy as np
import streamlit as st

from src.engine.model_engine import load_model_state

LABEL_DICT = {
    "Positif": 0,
    "Netral": 1,
    "Negatif": 2,
}

st.set_page_config(
    page_title="Stunting Sentiment Analysis",
    layout="wide",
)


def main():
    bert_berita, roberta_medsos = load_model_state()

    st.title("Stunting Sentiment Analysis")
    st.write(
        "This is a simple web app to predict sentiment analysis of stunting in Indonesia."
    )

    text = st.text_area("Enter text here")
    model_type = st.selectbox("Select model", ["BERT Berita", "RoBERTa Medsos"])

    if st.button("Predict"):
        if model_type == "BERT Berita":
            model = bert_berita
        elif model_type == "RoBERTa Medsos":
            model = roberta_medsos

        inputs = model.tokenize(text)
        _, preds = model.predict(inputs)

        sentiment = np.argmax(preds)
        sentiment = list(LABEL_DICT.keys())[list(LABEL_DICT.values()).index(sentiment)]

        st.write(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()

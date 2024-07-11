import rootutils

ROOT = rootutils.autosetup()

import numpy as np
import streamlit as st
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)


class ModelEngine:

    def __init__(self, checkpoint: str, type: str, device: str = "cpu"):
        super(ModelEngine, self).__init__()
        if type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
            self.model = BertForSequenceClassification.from_pretrained(checkpoint)
        elif type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
            self.model = RobertaForSequenceClassification.from_pretrained(checkpoint)
        else:
            raise ValueError("Invalid model type")
        self.device = device
        self.model.to(device)
        self.model.eval()

    def tokenize(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        return inputs.to(self.device)

    def predict(self, inputs: torch.Tensor):
        with torch.no_grad():
            logits = self.model(**inputs).logits

        preds = self.softmax(logits)

        return logits.to("cpu").numpy(), preds.to("cpu").numpy()

    def softmax(self, logits: torch.Tensor):
        return torch.nn.functional.softmax(logits, dim=1)


@st.cache_resource(show_spinner="⚙️ Loading model...")
def initialize_model():
    if "bert_model" not in st.session_state:
        st.session_state.model = ModelEngine(
            "hiseulgi/stunting-berita-sentiment", "bert", "cpu"
        )
    if "roberta_model" not in st.session_state:
        st.session_state.model = ModelEngine(
            "hiseulgi/roberta-stunting-medsos-sentiment", "roberta", "cpu"
        )

    return st.session_state.model


def load_model_state():
    if "bert_model" in st.session_state:
        bert_model = st.session_state.model
    else:
        bert_model = initialize_model()

    if "roberta_model" in st.session_state:
        roberta_model = st.session_state.model
    else:
        roberta_model = initialize_model()

    return bert_model, roberta_model


if __name__ == "__main__":
    print("Load model BERT Berita")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    bert = ModelEngine("hiseulgi/stunting-berita-sentiment", "bert", DEVICE)

    INPUT_TEXT = "Kasus stunting di Indonesia meningkat"
    print(f"Input text: {INPUT_TEXT}")

    print("Tokenizing input text")
    inputs = bert.tokenize(INPUT_TEXT)
    print(inputs)

    print("Predicting")
    logits, preds = bert.predict(inputs)
    print(logits)
    print(preds)

    pred_flat = np.argmax(preds, axis=1).flatten()
    print(pred_flat)

    print()

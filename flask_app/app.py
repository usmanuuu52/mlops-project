from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    text = "".join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub("\s+", " ", text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "usmanuuu52"
repo_name = "mlops-project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

app = Flask(__name__)


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    if not latest_version:
        raise ValueError(f"No versions found for model: {model_name}")
    return latest_version[0].version


model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(
        features.toarray(), columns=[str(i) for i in range(features.shape[1])]
    )
    result = model.predict(features_df)
    return render_template("index.html", result=result[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

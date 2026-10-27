"""Project 3: Spam SMS Detector (Classification)

Uses TF-IDF + Naive Bayes and Logistic Regression.

Dataset: download 'SMS Spam Collection' (UCI / Kaggle 'spam.csv') and place it
in this folder. Kaggle's spam.csv has columns v1 (label) and v2 (text) and
needs encoding='latin-1'. If no file is found, a tiny built-in sample is used
so you can see the pipeline run (results on it are NOT meaningful).

Run:  python spam_detector.py
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)


def load_data():
    if os.path.exists("spam.csv"):
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        return df
    print("[!] spam.csv not found - using a tiny demo sample.")
    spam = ["WINNER!! You have won a free prize, call now to claim",
            "Free entry in a weekly contest, text WIN to 80085",
            "URGENT! Your account is suspended, click the link to verify",
            "Congratulations, you've been selected for a cash reward",
            "Claim your free ringtone now, reply YES",
            "Get a loan approved instantly, no documents needed",
            "You have won a lottery of 1000000, send your bank details",
            "Limited offer! Buy one get one free, click here"]
    ham = ["Are we meeting for lunch tomorrow?",
           "Can you send me the notes from class",
           "I will be home by 8, save some dinner",
           "Happy birthday! Have a great day",
           "Did you finish the assignment for DSA?",
           "Call me when you reach the station",
           "Let's study together in the library tonight",
           "Thanks for helping me yesterday"]
    return pd.DataFrame({"label": ["spam"] * len(spam) + ["ham"] * len(ham),
                         "text": spam + ham})


def clean(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)


df = load_data()
df["text"] = df["text"].apply(clean)
df["target"] = (df["label"] == "spam").astype(int)
print(df["label"].value_counts(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.25, random_state=42, stratify=df["target"])

vec = TfidfVectorizer(stop_words="english")
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

models = {"Naive Bayes": MultinomialNB(),
          "Logistic Regression": LogisticRegression(max_iter=1000)}

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(f"=== {name} ===")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120)

# ---------- Try your own messages ----------
best = models["Logistic Regression"]
for msg in ["Congratulations! You won a free prize, click now",
            "Hey, are you coming to class tomorrow?"]:
    label = best.predict(vec.transform([clean(msg)]))[0]
    print(f"'{msg}' -> {'SPAM' if label else 'HAM'}")

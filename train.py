import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import jieba

def jieba_tokenizer(text):
    return list(jieba.cut(text))

df = pd.read_csv("data.csv", names=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.strip()
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer)),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

dump(pipeline.named_steps['clf'], 'classifier.pkl')
dump(pipeline.named_steps['tfidf'], 'vectorizer.pkl')

print("模型訓練與儲存完成")

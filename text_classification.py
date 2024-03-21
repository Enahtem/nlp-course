import numpy as np
import pandas as pd

df = pd.read_csv('moviereviews.tsv', sep='\t')
print(df.head())
# Dropping Empty String Reviews
print(df.isnull().sum())
df.dropna(inplace=True)
# Dropping Whitespace String Reviews
blanks = []
for index,label,review in df.itertuples():
    if type(review)==str:
        if review.isspace():
            blanks.append(index)
print(len(blanks), 'blanks: ', blanks)
df.drop(blanks, inplace=True)
print(len(df))
print("Label Values: ", df['label'].value_counts())
# Splitting Data
from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB()),])
# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC()),])


# Naïve Bayes Fitting, Predictions, Results
text_clf_nb.fit(X_train, y_train)
predictions = text_clf_nb.predict(X_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

# Linear SVC
text_clf_lsvc.fit(X_train, y_train)
predictions = text_clf_lsvc.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))


# Adding Stopwords (Ignored words)
stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

text_clf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),('clf', LinearSVC()),])
text_clf_lsvc2.fit(X_train, y_train)
predictions = text_clf_lsvc2.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))


# Individual Review Predictions
myreview = "Sunshine Serenade is a cinematic masterpiece that seamlessly weaves together a heartwarming narrative, breathtaking visuals, and an enchanting soundtrack. From the opening scene to the closing credits, the film captivates audiences with its compelling storytelling and genuine emotional resonance. The performances from the stellar cast are nothing short of outstanding, with each actor bringing their character to life with authenticity and depth."
print(text_clf_nb.predict([myreview]))
print(text_clf_lsvc.predict([myreview]))
print(text_clf_lsvc2.predict([myreview]))
myreview = "Shadows of Disappointment falls short of expectations, delivering a lackluster and disjointed cinematic experience. Despite the promising premise, the film struggles to find its footing, leaving audiences with a sense of confusion and dissatisfaction. The plot, which initially holds promise, meanders without a clear direction, leaving many unanswered questions by the time the credits roll."
print(text_clf_nb.predict([myreview]))
print(text_clf_lsvc.predict([myreview]))
print(text_clf_lsvc2.predict([myreview]))

import pandas as pd

npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])


from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7,random_state=42)
nmf_model.fit(dtm)


single_topic = nmf_model.components_[0]
top_word_indices = single_topic.argsort()[-10:]


for index in top_word_indices:
    print(tfidf.get_feature_names_out()[index])


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')

topic_results = nmf_model.transform(dtm)
npr['Topic'] = topic_results.argmax(axis=1)


print(npr.head(10))
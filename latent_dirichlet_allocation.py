import pandas as pd
npr = pd.read_csv('npr.csv')


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])


from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)
# This can take awhile, we're dealing with a large amount of documents!
LDA.fit(dtm)

len(cv.get_feature_names_out())# Stored words

# Random words
import random
for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names_out()[random_word_id])

print(len(LDA.components_))
print(len(LDA.components_[0]))
single_topic = LDA.components_[0]

# Returns the indices that would sort this array.
single_topic.argsort()

# Top 10 words for this topic:
top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(cv.get_feature_names_out()[index])


# Top 10 
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')

topic_results = LDA.transform(dtm)

print(topic_results[0].argmax())

print(npr.head(10))

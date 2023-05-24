from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# defining the category map with 4 categories - religion, autos, sport, electornics, space
category_map = {'talk.religion.misc': 'Religion', 'rec.autos':'Autos','rec.sport.hockey':'Hockey', 'sci.space':'Space'}

#create training dataset
training_data = fetch_20newsgroups(subset = 'train', categories = category_map.keys(), shuffle = True, random_state=5)

#build the count vectorizer and extract the term counts
vectorizer_count = CountVectorizer()
train_tc = vectorizer_count.fit_transform(training_data.data)
print("\n Dimensions of training data:", train_tc.shape)

#tfidf transformer is created as follows
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

#defining the test data:
input_data = [
    'Discovery was a space shuttle',
    'Hindu, Sikh, Christian all are religions',
    'We must have to drive safely',
    'Puck is a disk made of rubber',
    'Television, Microwave, Refrigrator all uses electricity'
]

classifier = MultinomialNB().fit(train_tfidf, training_data.target)

#transform the input data using count vectorizer:
input_tc = vectorizer_count.transform(input_data)

#now we will transform the vectorized data using the tdidf transformer
input_tfidf = tfidf.transform(input_tc)

#we will predict the output categories
predictions = classifier.predict(input_tfidf)

#the is generated as follows:
for sent, category in zip(input_data, predictions):
    print('\nInput Data: ', sent, '\nCategory: ', category_map[training_data.target_names[category]])


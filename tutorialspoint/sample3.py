from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

train, test, train_labels, test_labels = train_test_split(features, labels, test_size =0.40, random_state =42)
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

preds = gnb.predict(test)
print(preds)


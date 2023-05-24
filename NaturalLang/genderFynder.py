import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names 

# we need to extract the last N words from the input word
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}

#create the training data using labeled names (males as well females) available in NLTK
if __name__ == '__main__':
    male_list = [(name, 'male') for name in names.words('male.txt')]
    female_list = [(name, 'female') for name in names.words('female.txt')]
    data = (male_list + female_list)
    random.seed(5)
    random.shuffle(data)

    namesInput = ['Jack', 'Scarlet', 'Emily', 'John']

    #define number of samples used for train and test 
    train_sample = int(0.8 * len(data))

    #now we need to iterate via different lengths so that accuracy can be compared
    for i in range(1,6):
        print('\n Number of end letters: ', i)
        features = [(extract_features(n,i), gender) for (n, gender) in data]
        train_data, test_data = features[:train_sample], features[train_sample:]
        classifier = NaiveBayesClassifier.train(train_data)
        #accuracy of the classifier can be computed as follows
        accuracy_classifier = round(100 * nltk_accuracy(classifier, test_data))
        print('Accuracy = ' + str(accuracy_classifier) + '%')
        for name in namesInput:
            print(name, '==>', classifier.classify(extract_features(name, i)))





from csv import DictReader, DictWriter

import numpy as np
import nltk, re
from numpy import array
from nltk.stem import *


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'Label'
kTEXT_FIELD = 'Text'


class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7)

    def append_data_to_sentence(self, text):

        text = re.sub(r'[^\w\s]',' ',text)

        # pos = ''
        pos_tok = nltk.word_tokenize(text)
        # pos_li = nltk.pos_tag(pos_tok)
        # for ii in pos_li:
        #     pos = pos + ' ' + ii[1]
        POS_string=self.POS_converter(text)
        lemma_string=self.stemmer_unigram(text)
        length = ' ' + str(len(pos_tok))
        bigrams = self.word_ngram(text,2)
        result =  length + bigrams + lemma_string

        return result
        
    def POS_converter(self,text):
        POS_list=nltk.pos_tag(text)
        PoS=[item[1] for item in POS_list]
        string = ' '.join(PoS)
        return string
        
    def stemmer_unigram(self,token):
        stemmer = PorterStemmer()
        token=token.split()
        base = [stemmer.stem(t) for t in token]
        string=' '.join(base)
        return string

    def word_ngram(self,token,N):
        if N < 2: return 'please give a number bigger than one'
        else:
            token = token.split(' ')
            output = []
            for m in range(2,N+1):
                for i in range(len(token)-m+1):
                    output.append(token[i:i+m])
            o=[]
            for item in output:
                if len(item)>=2:
                    o.append(''.join(item))
                else:o.append(item[0])
            o=' '.join(o)
            return o         
        
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    dev_train = []
    dev_test = []
    y_dev_train = []
    y_dev_test = []
    count = 0

    full_train = []
    full_test = []
    y_full = []

    for ii in train:
        count += 1
        if count % 5 == 0:
            # Appends feature stem and category key
            dev_test.append(feat.append_data_to_sentence(ii['Text']))
            y_dev_test.append(ii['Label'])
        else:
            # Appends feature stem and category key
            dev_train.append(feat.append_data_to_sentence(ii['Text']))
            y_dev_train.append(ii['Label'])
        
        # Full
        # full_train.append(feat.append_data_to_sentence(ii['sentence'] + ' ' + ii['page'] + ' ' + ii['trope']))
        # y_full.append(ii['spoiler'])

    # Full
    # for ii in test:
    #     full_test.append(feat.append_data_to_sentence(ii['sentence'] + ' ' + ii['page'] + ' ' + ii['trope']))

    x_dev_train = feat.train_feature(x for x in dev_train)
    x_dev_test = feat.test_feature(x for x in dev_test)

    # Full
    # x_train = feat.train_feature(x for x in full_train)
    # x_test = feat.test_feature(x for x in full_test)

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_dev_train, y_dev_train)

    # Full
    # lr.fit(x_train, y_full)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_dev_test)
    # predictions = lr.predict(x_test)


    # Test the classfier 
    right = 0
    total = len(dev_test)
    for ii in range(len(y_dev_test)):
        if predictions[ii] == y_dev_test[ii]:
            right += 1
    print("Accuracy on dev: %f" % (float(right) / float(total)))

    # Full
    # o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    # o.writeheader()
    # for ii, pp in zip(range(len(full_test)), predictions):
    #     d = {'id': ii, 'spoiler': pp}
    #     o.writerow(d)

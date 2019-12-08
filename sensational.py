import numpy as np
import pandas as pd
import warnings
import pickle
import string
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def prediction(xtest, ytest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/sensational_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    return np.mean(loadData.predict(xtest) == ytest)

def processFakeNews(fnews):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    pcCount=[]
    capCount=[]
    digCount=[]
    lenCount=[]
    profanCount=[]
    sensphrCount=[]
    for x in fnews['Statement']:
       pcCount.append(sum(1 for c in x if c=="!" or c=="?"))
       capCount.append(sum(1 for c in x if c.isupper()))
       digCount.append(sum(1 for c in x if c.isdigit()))
       lenCount.append(len(x))
       sensphrCount.append(sensphrasedetect(x))
#    for x in test_reviewed_docs:
#       try:
#          profanCount.append(float(len([w for w in x if w.lower() in PROFANITY]))/len(x))
#       except:
#          profanCount.append(0)

    fnews['puncCount']=pcCount
    fnews['capCount']=capCount
    fnews['digCount']=digCount
    fnews['lenCount']=lenCount
#    fnews['profanCount']=profanCount
    fnews['profanCount']=0
    fnews['sensPhrCount']=sensphrCount
    fnews 

def buildSensationalCol(f_news):
    savedModel = "/content/gdrive/My Drive/Drifters/Models/sensationalism.model"
    sensationCol=[]
    model= Doc2Vec.load(savedModel)
    for row in valid_news['headline_text']:
        test_data = word_tokenize(row.lower())
        v1 = model.infer_vector(test_data)
        similar_doc = model.docvecs.most_similar([v1])
        sensationCol.append(similar_doc[0][0])
    sensationCol=list(map(int, sensationCol))
    f_news['sensationCol']=1
    f_news
        
class sensational:
    def __init__(self, fnews):
        self.f_news = processFakeNews(fnews)
        self.x_test = buildSensationalCol(self.f_news)
        self.y_test = self.x_test['Label'].map({'false':0,'true': 1,'barely-true':0,'half-true':1,'mostly-true':1,'pants-fire':0})
    def predict(self):
        return prediction(self.x_test, self.y_test)

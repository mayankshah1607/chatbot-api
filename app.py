from flask import Flask,request,render_template
import keras.models
from load import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import nltk
import json
import flask


port = int(os.environ['PORT'])
data = pd.read_csv('./data/data.csv')

for index,row in data.iterrows():
    if row['Action'] == 'Others':
        data = data.drop([index])

vocab = []
for index,row in data.iterrows():
    tokens = nltk.word_tokenize(row['Query'])
    for i in tokens:
        if not i in vocab:
            vocab.append(i)
vocab.append('UNK')
vocab.append('PAD')

n_words = len(vocab)
actions = list(data['Action'].unique())
n_actions = len(actions)

action_index_1 = {}
action_index_2 = {}

for i,v in enumerate(actions):
    action_index_1[i] = v
    action_index_2[v] = i

def get_index_matrix(sentence):
    matrix = []
    w = nltk.word_tokenize(sentence)
    for i in w:
        if i in vocab:
            matrix.append(vocab.index(i))
        else :
            matrix.append(vocab.index('UNK'))
    x = pad_sequences(maxlen=18, sequences=[matrix], padding="post", value=vocab.index('PAD'))
    return x


app = Flask(__name__)

global model,graph
model,graph = init()



def get_prediction(query):
    a = nltk.word_tokenize(query)
    a = [i.lower() for i in a]
    a = [i for i in a if i.isalpha()]
    sentence = ' '.join(a)
    x = get_index_matrix(sentence)
    prediction = model.predict([x])[0]
    ans = np.argmax(prediction)
    score = round(max(prediction)*100,2)
    return action_index_1[ans],score


@app.route('/predict',methods=['GET','POST'])
def predict():

    query = json.loads(request.data)['query']
    with graph.as_default():

        action,score = get_prediction(query)
        return flask.jsonify({'Score': score,'Action': action})


    
    


if __name__ == '__main__':
    app.run(host='0.0.0.0',port = port, debug = True)
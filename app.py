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
from flask_cors import CORS, cross_origin

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

messages = {
    'get_event_fees': 'The cost of this event is $100.',
    'is_refundable': 'This event is not refundable.',
    'get_registration_date': 'The registration begins from 12th June, 2019.',
    'get_payment_method': 'You can pay by any card and event PayTm or Tez.',
    'get_prizes': 'The winner of this event gets a cash prize of $100,000.',
    'get_discounts': 'Sorry, there are no discounts yet!',
    'greet': 'Hello there! Ask me anything about this event.',
    'show_schedule': 'The event starts at 1300Hrs, followed by a break at 1500hrs. The final ceremony is at 1600hrs',
    'get_event_date': 'This event is happening on 23rd July, 2019.',
    'get_event_time': 'The event begins at around 1300Hrs',
    'show_accomodation': 'We have booked hotels at GRT Hotels.',
    'show_speakers': 'Some great speakers for this event are Bill Gates, Elon Musk and Jeff Bezos',
    'speaker_details_extra': 'The speakers will be staying for a few hours after the event to click pictures!',
    'show_food_arrangements': 'We have arranged for food from McDonalds',
    'get_distance': 'From your current location, the venue is 12kms away.',
    'get_location': 'The event is happening in Anna Auditorium, VIT Vellore',
    'show_contact_info': 'You can contact Mr.Mayank Shah - 9937162937',
    'about_chatbot': 'I am a smart Question answering chatbot made by Mayank Shah.'
}

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

CORS(app)

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
        return flask.jsonify({'Score': score,'Action': action, 'Message': messages[action]})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = port, debug = True)
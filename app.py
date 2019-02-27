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
    'get_event_fees': 'For IEEE members the registration fees is Rs.399, and for non-IEEE members it is Rs.499. Both include Techloop sessions and Hack-a-thon.',
    'is_refundable': 'Sorry, the event fee is non-refundable.',
    'get_registration_date': 'The registrations have already begun. Register now at ith.ieeevit.com/register , and follow our facebook page to stay updated!',
    'get_payment_method': 'On registering at ith.ieeevit.com/register , you will be redirected to the payment portal.',
    'get_prizes': 'Haha, that\'s a surprise! ;)',
    'get_discounts': 'Sorry, there are no discounts yet!',
    'greet': 'Hello there! Ask me anything about this event. Eg: When is the event? What are the registration fees?',
    'show_schedule': 'The itinerary has not been finalized yet. We will get back to you soon!',
    'get_event_date': 'This event is happening on 17th and 18th of March, 2019.',
    'get_event_time': 'Sorry, the exact timings have not been finalized yet!',
    'show_accomodation': 'Sorry, we do not have information regarding accommodation. You may contact Druvang Wagh - 8369091192',
    'show_speakers': 'As of now, we have mentors and speakers from Microsoft and Google.',
    'speaker_details_extra': 'AS of now, we have mentors and speakers from Microsoft and Google.',
    'show_food_arrangements': 'Haha, you seem to be hungry. But, sorry, we have not yet finalized the food arrangements.',
    'get_distance': 'Hmmm, check google maps?',
    'get_location': 'The event is happening in VIT Vellore. Exact venue will be informed soon.',
    'show_contact_info': 'You may contact Druvangh Wagh - 8369091192',
    'about_chatbot': 'I am Lexie, a smart assistant that can answer all your queries regarding ITC 2019. What would you like to know?'
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
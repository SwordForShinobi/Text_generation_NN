import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pandas as pd

data = pd.read_csv('Model/amazon_reviews.csv')

model = tf.keras.models.load_model('Model/Peter_model.h5')


def preprocess(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '', text_input)
    return output.lower().strip()


data['reviewText'] = data.reviewText.map(preprocess)
corpus_cleaned = data['reviewText'].astype(str).values.tolist()

sentences = []
sentence_length = []
num_words = 12

for item in corpus_cleaned:
    word_list = item.split()
    sentences.append(item)
    number_of_words = len(word_list)
    sentence_length.append(number_of_words)

sentences = pd.DataFrame(sentences)
sentence_length = pd.DataFrame(sentence_length)

sentences = sentences.merge(sentence_length, left_index=True, right_index=True)

sentences['number_of_words'] = sentences['0_y']
sentences['sentence'] = sentences['0_x']
sentences = sentences[['sentence', 'number_of_words']]
sentences['number_of_words'] = sentences['number_of_words'].astype(int)

sentences = sentences[sentences['number_of_words'] == num_words]
corpus_cleaned = sentences['sentence'].values.tolist()
corpus = corpus_cleaned

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

# create input sequences using list of tokens
input_sequences = []

for review in corpus:
    token_list = tokenizer.texts_to_sequences([review])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

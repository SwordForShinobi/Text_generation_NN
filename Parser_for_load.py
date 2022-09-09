import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from Model.Model_load import tokenizer, model, input_sequences

parser = argparse.ArgumentParser()
parser.add_argument('seed_text', type=str)
parser.add_argument('size', type=int)
args = parser.parse_args()


def text_generator(seed_text, size):
    for _ in range(size):
        tokens = tokenizer.texts_to_sequences([seed_text])[0]
        max_sequence_len = max([len(x) for x in input_sequences])
        tokens = pad_sequences([tokens], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(tokens, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                output_word = word
                break
        seed_text += " " + output_word
    seed_text = (seed_text + ".").capitalize()
    print(seed_text)


text_generator(args.seed_text, args.size)

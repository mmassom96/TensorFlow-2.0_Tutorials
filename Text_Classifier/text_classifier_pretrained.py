import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen = 256)
# test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen = 256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")

# with open("review_django_unchained.txt", encoding="utf-8") as f:
with open("review_django_unchained.txt") as f:
    review = ""
    for line in f.readlines():
        review = review + line.lower()
    review = review.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("\n", " ").replace("/", " ").replace("-", " ").strip().split(" ")
    encode = review_encode(review)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen = 256)
    predict = model.predict(encode)
    # print(review)
    # print(encode)
    print("\n" + f.name.replace("review_" ,"").replace("_", " ").replace(".txt", ""))
    print("Prediction: " + str(predict[0]))

with open("review_batman_&_robin.txt") as f:
    review = ""
    for line in f.readlines():
        review = review + line.lower()
    review = review.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("\n", " ").replace("/", " ").replace("-", " ").strip().split(" ")
    encode = review_encode(review)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen = 256)
    predict = model.predict(encode)
    # print(review)
    # print(encode)
    print("\n" + f.name.replace("review_" ,"").replace("_", " ").replace(".txt", ""))
    print("Prediction: " + str(predict[0]))

with open("review_fight_club.txt") as f:
    review = ""
    for line in f.readlines():
        review = review + line.lower()
    review = review.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("\n", " ").replace("/", " ").replace("-", " ").strip().split(" ")
    encode = review_encode(review)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen = 256)
    predict = model.predict(encode)
    # print(review)
    # print(encode)
    print("\n" + f.name.replace("review_" ,"").replace("_", " ").replace(".txt", ""))
    print("Prediction: " + str(predict[0]))

with open("review_son_of_mask.txt") as f:
    review = ""
    for line in f.readlines():
        review = review + line.lower()
    review = review.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("\n", " ").replace("/", " ").replace("-", " ").strip().split(" ")
    encode = review_encode(review)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen = 256)
    predict = model.predict(encode)
    # print(review)
    # print(encode)
    print("\n" + f.name.replace("review_" ,"").replace("_", " ").replace(".txt", ""))
    print("Prediction: " + str(predict[0]))
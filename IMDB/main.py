import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn as sk

data = keras.datasets.imdb

(train_data, train_lbls), (test_data, test_lbls) = data.load_data(num_words=40000)

# print(train_data[0])

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(test_data[0]))
# this function will return the decoded (human readable) reviews

model = keras.Sequential()
model.add(keras.layers.Embedding(40000, 12))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(12, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_lbls[:10000]
y_train = train_lbls[10000:]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_lbls[:10000]
y_train = train_lbls[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_lbls)
print(results)

# Saving and Loading the Models
#model.save("model.h5")


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		#print(line)
		print(encode)
		print(predict[0])

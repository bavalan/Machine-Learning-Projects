import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plots

data = keras.datasets.fashion_mnist

(train_imgs, train_lbls), (test_imgs, test_lbls) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Divide by 255.0 because that is the highest pixel value
# This was done to shrink down the data, will be easier to work with
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Show the [i]th image, cmap(shows the original color)
# plots.imshow(train_imgs[5], cmap=plots.cm.binary)
# plots.show()

# relu-rectified linear unit
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs- how many time the model going to see the same info
model.fit(train_imgs, train_lbls, epochs=5)

#test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
#print('\nTest accuracy:', test_acc)

#Gives us a group of prediction
predict = model.predict([test_imgs])

#Gives us largest neuron value and its index
#(class_names[np.argmax(predict[0])])
for x in range(5):
    plots.grid(False)
    plots.imshow(test_imgs[x], cmap=plots.cm.binary)
    plots.xlabel("Result: " +class_names[test_lbls[x]])
    plots.title("Prediction: "+class_names[np.argmax(predict[x])])
    plots.show()

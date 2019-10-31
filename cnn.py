# Reference: https://github.com/MrKhan0747/Blood-Cell-Subtypes-Classification

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from data import create_data, separate_features_and_label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

from sklearn.metrics import accuracy_score

train_data = create_data()
test_data = create_data(train=False)

train_X, train_y = separate_features_and_label(train_data)
test_X, test_y = separate_features_and_label(test_data)

categorical_train_y = to_categorical(train_y)
categorical_test_y = to_categorical(test_y)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(train_X.shape[1:]), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(64, activation="relu"))

model.add(Dense(4, activation="softmax"))

model.summary()

adam = optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

hist = model.fit(train_X, train_y, epochs=50, batch_size=128, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_X, test_y)

# train and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# train and validation accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# model prediction

y_pred = model.predict_classes(test_X)
print(y_pred)

# for i in range(10):
#     print("Actual=%s, Predicted=%s" % (test_y[i], y_pred[i]))

# accuracy_score
accuracy_score(test_y, y_pred)

# learning rate
print("Learning Rate: " + str(K.eval(model.optimizer.lr)))

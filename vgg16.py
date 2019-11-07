# Reference: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from data import create_data, separate_features_and_label

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_data, validation_data = train_test_split(create_data(), test_size=0.2, random_state=42)
test_data = create_data(train=False)

train_X, train_y = separate_features_and_label(train_data)
validation_X, validation_y = separate_features_and_label(validation_data)
test_X, test_y = separate_features_and_label(test_data)

categorical_train_y = to_categorical(train_y)
categorical_test_y = to_categorical(test_y)

# model = Sequential()
#
# model.add(Conv2D(input_shape=(train_X.shape[1:]), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.1))
#
# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(units=4096, activation="relu"))
# model.add(Dense(units=4096, activation="relu"))
# model.add(Dense(units=4, activation="softmax"))
#
# model.summary()
#
# adam = optimizers.Adam(lr=0.01)
# model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
#
# hist = model.fit(train_X, categorical_train_y, epochs=50, batch_size=32, validation_split=0.2)

vgg16_model = VGG16()

model = Sequential()

for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(4, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit_generator(train_data, steps_per_epoch=8,
                           validation_data=validation_data, validation_steps=8,
                           epochs=5, verbose=2)

predictions = model.predict_generator(test_data, steps=1, verbose=0)
test_loss, test_acc = model.evaluate(test_X, categorical_test_y)

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

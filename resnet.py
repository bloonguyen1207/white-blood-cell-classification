# Reference: https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb

import matplotlib.pyplot as plt
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical

from data import IMG_SIZE, CATEGORIES, create_data, separate_features_and_label

train_data = create_data()
test_data = create_data(train=False)

train_X, train_y = separate_features_and_label(train_data)
test_X, test_y = separate_features_and_label(test_data)

categorical_train_y = to_categorical(train_y)
categorical_test_y = to_categorical(test_y)

base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model.output
x = GlobalMaxPool2D()(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.7)(x)
predictions = Dense(len(CATEGORIES), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

adam = optimizers.Adam(lr=0.00005)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_X, categorical_train_y, epochs=20, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_X, categorical_test_y)
print("Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_acc))

model.summary()

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

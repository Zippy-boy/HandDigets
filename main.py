import os
import cv2  # commutator vision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist  # The hand number nad what it is
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # split to training data and test data || x is the pixle data y is what Number

x_train = tf.keras.utils.normalize(x_train, axis=1)  #  make all valus 0 to 1 instaed of 1-255
x_test = tf.keras.utils.normalize(x_test, axis=1)




model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # makes grid of pixles into one big line of 7840 pixles
model.add(tf.keras.layers.Dense(236, activation='relu'))  # rectify linior unit
model.add(tf.keras.layers.Dense(10, activation="softmax"))   # output layer || softmax = pick the most confident nuron
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)  # Train model || epoch = how many time brain sees same data
model.save('HandWriteModel.model')



#model = tf.keras.models.load_model('HandWriteModel')

image_number = 1
while  os.path.isfile(f"DigetsByMe\\diget{image_number}.png"):
    try:
        img = cv2.imread(f"DigetsByMe\\diget{image_number}.png")[:,:,0]  # rgb?
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("Img is probable not 28 by 28")
    finally:
        image_number += 1



loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

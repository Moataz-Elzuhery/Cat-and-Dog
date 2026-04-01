import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("best_model.h5")

path = input("Enter image path: ")
img = cv2.imread(path)
img = cv2.resize(img, (150,150))
img = img / 255.0

img = np.reshape(img, (1,150,150,3))

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
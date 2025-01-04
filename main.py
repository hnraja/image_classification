import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import models
import os

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog',
               'Frog', 'Horse', 'Ship', 'Truck']

# Load model
model = models.load_model("model.keras")

for img in os.listdir("images"):
    # Load images
    file = f"images/{img}"
    img = cv.imread(file)
    # imread loads images in BGR, we want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # change image resolution (model trained on low-res)
    img = cv.resize(img, (32,32))

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    print(f"Prediction is {class_names[index]}")
    plt.show()
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load model
model = models.load_model("model.keras")

for file in ['horse.jpg', 'plane.jpg', 'car.jpg', 'deer.jpg']:
    # Load images
    img = cv.imread(file)
    # imread loads images in BGR, we want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    print(f"Prediction  is {class_names[index]}")
    plt.show()
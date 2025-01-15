from keras import datasets, layers, models
from timeit import default_timer as timer
import csv

# Load data
(train_img, train_lab), (test_img, test_lab) = datasets.cifar10.load_data()
train_img, test_img = train_img/ 255, test_img / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

max_accuracy = 0

# Convolutional layer filters for features in images
# Max pooling layer reduces image to essential information
# Output dense layer scales the result for probabilities

# finding optimal combination of functions and save to file
with open("performance/performance.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Activation", "Optimizer", "Accuracy", "Loss", "Time"])
    for activation in ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu',
                       'exponential', 'leaky_relu', 'relu6']:
        for optimizer in ['SGD', 'rmsprop', 'adam', 'adamw', 'adadelta', 'adagrad', 'nadam', 'ftrl', 'lion']:
            # initiliaze
            start = timer()
            model = models.Sequential()

            # add layers
            model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add((layers.Conv2D(64, (3, 3), activation=activation)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add((layers.Conv2D(64, (3, 3), activation=activation)))

            # flatten
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation=activation))
            model.add(layers.Dense(10, activation='softmax'))

            # fit
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=10)

            end = timer()

            # save model metrics
            loss, accuracy = model.evaluate(test_img, test_lab)
            writer.writerow([activation, optimizer, accuracy, loss, end - start])

            # update best model if necessary
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                model.save("model.keras")

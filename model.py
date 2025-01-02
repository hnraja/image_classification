from keras import datasets, layers, models

# Load data
(train_img, train_lab), (test_img, test_lab) = datasets.cifar10.load_data()
train_img, test_img = train_img/ 255, test_img / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

max_accuracy = 0

# Convolutional layer filters for features in images
# Max pooling layer reduces image to essential information
# Output dense layer scales the result for probabilities


# finding optimal number of pooling layers
for i in range(1,4):
    # initialize
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))

    # add i layers
    for j in range(i):
        model.add(layers.MaxPooling2D((2, 2)))
        model.add((layers.Conv2D(64, (3,3), activation='relu')))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # fit model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=10)

    # evaluate model
    loss, accuracy = model.evaluate(test_img, test_lab)
    print(f"Model with {i} pooling layers has {accuracy:.2%} accuracy")

    # update best model if necessary
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model.save("model.keras")


import keras

inpt = keras.Input()
modl = keras.Sequential()
data = keras.datasets.cifar10.load_data()

#cifar10.load_data()
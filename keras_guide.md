# Models
## `Sequential`
```python
from keras import models
model = models.Sequential()
```

* special case of models
* model is just stack of 1 input, 1 output layers
* `model.add()` is used to add new layers
* `model.pop()` removes the last layer

### Method 1: with an initial `Input`
```python
from keras import models, layers
model = models.Sequential()
model.add(layers.Input(shape=(16,)))
model.add(layers.Dense(8))
```
* model gets built continuously as you add layers
* i.e., `model.weights` is automatically created

### Method 2: delayed-build pattern
```python
from keras import models, layers
model = models.Sequential()
model.add(layers.Dense(8))
model.add(layers.Dense(4))
# model.weights not created yet
model.build((None, 16)) # manually creates model.weights
```
* no `Input`
* `model.weights` not created till first call
* can build manually using `model.build()`
* when no input shape specified, model built on first call
    * `fit` trains model for fixed # of epochs
    * `eval`
    * `predict` generate output predictions for input samples
    * model call on input data

## `Input` 
```python
from keras import layers, models
inputs = layers.Input(shape=(37,))
x = layers.Dense(32, activation="relu")(inputs)
outputs = layers.Dense(5, activation="softmax")(x)
model = models.Model(inputs=inputs, outputs=outputs)
```
* starting from `Input`
* chain layers
* create model from inputs and outputs



source: https://keras.io/api/

# object_detection_classification

Limitations
* can only identify images of the following
    * planes
    * cars
    * birds
    * cats
    * deer
    * dogs
    * frogs
    * horses
    * ships
    * trucks
* since model is trained on built-in keras datasets containing only these objects 


File Tree
* model.py 
    * code used to create model.keras
    * fits multiple models and selects best model based on accuracy
* main.py 
    * uses model.keras to identify the images in ./images
* images/
    * image source: https://pixabay.com/
    * contains images of 

Learning Objectives
* Keras datasets
* Keras models
* Keras layers
